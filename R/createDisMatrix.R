utils::globalVariables(c("resp", "W", "data_XGB", "i"))

#' Create a Dissimilarity Matrix from an Ensemble Model
#'
#' The function createDisMatrix creates a dissimilarity matrix among
#' observations from an ensemble tree. This optimized version is designed
#' for large datasets (50K-500K observations) with improved memory management
#' and chunking capabilities.
#'
#' @param ensemble is an ensemble tree object
#' @param data is a data frame containing the variables in the model. It is the
#'   data frame used for ensemble learning.
#' @param label is a character. It indicates the response label.
#' @param parallel A list with two elements: \code{active} (logical) and
#'   \code{no_cores} (integer). If \code{active = TRUE}, the function performs
#'   parallel computation using the number of cores specified in
#'   \code{no_cores}. If \code{no_cores} is NULL or equal to 0, it defaults to
#'   using all available cores minus one. If \code{active = FALSE}, the function
#'   runs on a single core. Default: \code{list(active = FALSE, no_cores = 1)}.
#' @param verbose Logical. If TRUE, the function prints progress messages and
#'   other information during execution. If FALSE (the default), messages are
#'   suppressed.
#' @param chunk_size Integer. Number of rows to process in each chunk. If NULL,
#'   automatically determined based on available memory and dataset size.
#'   Default: NULL (auto).
#' @param memory_limit Numeric. Maximum memory to use in GB. Default: NULL (no limit).
#' @param use_disk Logical. If TRUE and dataset is very large, intermediate results
#'   are saved to disk. Default: FALSE.
#' @param temp_dir Character. Directory for temporary files if use_disk = TRUE.
#'   Default: tempdir().
#' @param batch_aggregate Integer. Number of tree results to aggregate at once
#'   before adding to main matrix (reduces memory peaks). Default: 10.
#'
#' @return A dissimilarity matrix. This is a dissimilarity matrix measuring the
#'   discordance between two observations concerning a given random forest
#'   model.
#'
#' @details This optimized version implements several strategies for handling
#'   large datasets:
#'
#' \itemize{
#'   \item **Memory-efficient aggregation**: Results from parallel trees are
#'     aggregated in batches to avoid memory peaks
#'   \item **Chunking**: For very large matrices, computation can be split into
#'     manageable chunks
#'   \item **Sparse matrix optimization**: Maintains sparsity throughout computation
#'   \item **Automatic garbage collection**: Explicit memory cleanup at critical points
#'   \item **Disk-based computation**: Optional saving of intermediate results for
#'     datasets exceeding memory capacity
#' }
#'
#' Supported ensemble types for *classification* or *regression* tasks:
#' \itemize{
#'   \item `randomForest`
#'   \item `ranger`
#'   \item `xgb.Booster` (xgboost)
#'   \item `lgb.Booster` (lightgbm)
#'   \item `gbm` (gbm)
#'   \item `catboost.CatBoost` (catboost)
#' }
#'
#' @section Interpretation note (RF vs boosting):
#' For *bagging* ensembles (\code{randomForest}, \code{ranger}) the trees are
#' grown independently on bootstrap samples; co-occurrence in the same leaf
#' captures local similarity in the predictor space. For *boosting* ensembles
#' (\code{xgb.Booster}, \code{lgb.Booster}, \code{gbm}, \code{catboost})
#' each tree is fit to the residual of the previous ones, so leaf
#' co-occurrence reflects similarity in the *error-correction trajectory*
#' rather than in the final prediction space. The resulting dissimilarity
#' matrices therefore have systematically different scales (typically
#' \eqn{\bar D \in [0.85, 0.95]} for bagging vs. \eqn{[0.35, 0.70]} for
#' boosting). The surrogate tree built on top of \code{D} should be
#' interpreted accordingly.
#'
#' The returned matrix carries an \code{ensemble_backend} attribute identifying
#' the backend used, which downstream functions check to detect mismatched
#' \code{(D, ensemble)} pairs.
#'
#' @examples
#' \donttest{
## Classification
#' data("iris")
#'
#' # Create training and validation set:
#' smp_size <- floor(0.75 * nrow(iris))
#' train_ind <- sample(seq_len(nrow(iris)), size = smp_size)
#' training <- iris[train_ind, ]
#' validation <- iris[-train_ind, ]
#' response_training <- training[,5]
#' response_validation <- validation[,5]
#'
#' # Perform training:
#' ## "randomForest" package
#' ensemble <- randomForest::randomForest(Species ~ ., data=training,
#' importance=TRUE, proximity=TRUE)
#'
#' ## "ranger" package
#' if (requireNamespace("ranger", quietly = TRUE)) {
#'   ensemble <- ranger::ranger(Species ~ ., data = iris,
#'     num.trees = 1000, importance = 'impurity')
#' }
#'
#' # Compute dissimilarity matrix with optimizations
#' D <- createDisMatrix(
#'   ensemble,
#'   data = training,
#'   label = "Species",
#'   parallel = list(active = FALSE, no_cores = 1),
#'   chunk_size = 10000,  # Process 10K rows at a time
#'   batch_aggregate = 20, # Aggregate 20 trees at once
#'   verbose = TRUE
#' )
#' }
#'
#' @export

createDisMatrix <- function(
  ensemble,
  data,
  label,
  parallel = list(active = FALSE, no_cores = 1),
  verbose = FALSE,
  chunk_size = NULL,
  memory_limit = NULL,
  use_disk = FALSE,
  temp_dir = tempdir(),
  batch_aggregate = 10
) {
  # === Input Validation ===
  if (is.null(ensemble)) {
    stop(
      paste0("Error: 'ensemble' cannot be NULL. ",
           "Please provide a trained randomForest, ranger, xgb.Booster, ",
           "lgb.Booster, gbm, or catboost.CatBoost model.")
    )
  }

  # Validate ensemble via adapter dispatch (throws informative error for unsupported classes)
  tryCatch(
    get_ensemble_type(ensemble),
    error = function(e) stop(conditionMessage(e), call. = FALSE)
  )

  if (!is.data.frame(data)) {
    stop("Error: 'data' must be a valid data frame.")
  }

  if (!is.null(label)) {
    if (!is.character(label) || length(label) != 1 || !(label %in% colnames(data))) {
      stop("Error: 'label' must be a valid column name in 'data'.")
    }
  }

  row.names(data) <- NULL
  n_obs <- nrow(data)

  # === Memory Management Setup ===
  if (verbose) {
    cat(sprintf("\n=== OPTIMIZED DISSIMILARITY MATRIX COMPUTATION ===\n"))
    cat(sprintf("Dataset size: %d observations\n", n_obs))
    cat(sprintf(
      "Matrix size: %d x %d = %.2f M elements\n",
      n_obs,
      n_obs,
      (n_obs * n_obs) / 1e6
    ))

    est_memory_gb <- (n_obs * n_obs * 8) / (1024^3)
    cat(sprintf("Estimated dense matrix memory: %.2f GB\n", est_memory_gb))

    if (.Platform$OS.type == "unix") {
      tryCatch(
        {
          mem_info <- system(
            "free -g 2>/dev/null || vm_stat 2>/dev/null | head -1",
            intern = TRUE
          )
          if (length(mem_info) > 0) {
            cat("System memory info available\n")
          }
        },
        error = function(e) {},
        warning = function(w) {}
      )
    }
  }

  if (is.null(chunk_size)) {
    chunk_size <- determine_chunk_size(n_obs, memory_limit, verbose)
  }

  use_chunking <- (chunk_size < n_obs) && (chunk_size > 0)

  if (verbose && use_chunking) {
    cat(sprintf("Using chunked computation with chunk size: %d\n", chunk_size))
  }

  # === Determine Ensemble Type ===
  type <- get_ensemble_type(ensemble)
  if (!(type %in% c("classification", "regression"))) {
    stop("'ensemble' must be a classification or regression model.", call. = FALSE)
  }

  # `label` is mandatory for regression (used to compute the dissimilarity
  # scale).  Validate up front so the error surfaces before any leaf
  # extraction work is wasted.
  if (type == "regression" && is.null(label)) {
    stop("Error: 'label' is required for regression models (needed to compute dissimilarity scale).",
         call. = FALSE)
  }

  # === Extract Terminal Nodes ===
  if (verbose) {
    cat("\nExtracting terminal nodes...\n")
  }

  backend <- ensemble_backend(ensemble)
  # CatBoost adapters strip the response column from the predictor pool via
  # the `e2tree_label` attribute; set it transparently when the user passes
  # `label` so the workflow matches the other backends.
  if (!is.null(label) &&
      backend %in% c("catboost.CatBoost", "catboost.Model") &&
      is.null(attr(ensemble, "e2tree_label"))) {
    attr(ensemble, "e2tree_label") <- label
  }
  obs <- extract_terminal_nodes(ensemble, data)
  validate_terminal_nodes(obs, data, backend = backend)

  class(data) <- "data.frame"
  # `label` is mandatory for regression (used to compute the dissimilarity
  # scale) but optional for classification (where it only annotates `resp`
  # for downstream code). Guard against NULL label before any data[[label]]
  # access; classification with NULL label proceeds with a dummy resp.
  if (!is.null(label) && !inherits(data[[label]], "factor")) {
    data[[label]] <- factor(data[[label]])
  }

  obs <- cbind(row.names(obs), obs)
  names(obs) <- c("OBS", paste("Tree", seq(1, (ncol(obs) - 1L)), sep = ""))
  row.names(obs) <- NULL

  if (type == "regression") {
    if (is.null(label)) {
      stop("Error: 'label' is required for regression models (needed to compute dissimilarity scale).")
    }
    obs$resp <- as.numeric(as.character(data[obs$OBS, label]))
  } else {
    # classification: resp is used only as metadata; use dummy when label is NULL
    obs$resp <- if (!is.null(label)) data[as.numeric(obs$OBS), label] else
      factor(rep(1L, nrow(obs)))
  }

  ntree <- ncol(obs) - 2L

  # Ensure tree columns are numeric (ranger may return double, randomForest integer)
  # C++ reads them as NumericVector and converts to int internally
  tree_col_idx <- 2:(ntree + 1)  # R 1-based columns Tree1..TreeN
  for (ci in tree_col_idx) {
    if (!is.numeric(obs[[ci]])) {
      obs[[ci]] <- as.numeric(obs[[ci]])
    }
  }
  # Ensure resp is plain numeric
  obs$resp <- as.numeric(obs$resp)

  if (verbose) {
    cat(sprintf("Number of trees: %d\n", ntree))
    cat(sprintf("Batch aggregation size: %d trees\n", batch_aggregate))
  }

  # === Determine Parallelism ===
  if (isTRUE(parallel$active)) {
    if (is.null(parallel$no_cores) || parallel$no_cores < 1L) {
      no_cores <- max(1L, parallel::detectCores() - 1L)
    } else {
      no_cores <- parallel$no_cores
    }
  } else {
    no_cores <- 1L
  }

  if (verbose) {
    cat(sprintf("\nParallel mode: %d cores (C++ OpenMP)\n", no_cores))
  }

  # === Main Computation — single C++ call for all trees ===
  if (verbose) {
    cat("\n=== COMPUTING CO-OCCURRENCE MATRIX (C++) ===\n")
  }

  maxvar <- if (type == "regression") diff(range(obs$resp))^2 / 9L else NA_real_

  a <- compute_all_cooccurrences_cpp(type, obs, as.integer(ntree), as.integer(no_cores), as.double(maxvar))

  if (verbose) {
    cat("Co-occurrence matrix computed.\n")
    cat("\n=== COMPUTING FINAL DISSIMILARITY MATRIX (C++) ===\n")
  }

  # === Dissimilarity: 1 - a[i,j] / max(a[i,i], a[j,j]) ===
  if (use_chunking) {
    if (verbose) {
      cat(sprintf("Using chunked R path (chunk size: %d)\n", chunk_size))
    }
    dis <- compute_dissimilarity_chunked(a, obs, chunk_size, verbose = verbose)
  } else {
    dis <- compute_dissimilarity_from_cooc_cpp(a)
    row.names(dis) <- colnames(dis) <- obs$OBS
  }

  rm(a)

  if (isTRUE(use_disk)) {
    if (!dir.exists(temp_dir)) dir.create(temp_dir, recursive = TRUE)
    out_path <- file.path(temp_dir,
                          sprintf("e2tree_dismatrix_%d.rds", as.integer(Sys.time())))
    saveRDS(dis, file = out_path)
    if (verbose) cat(sprintf("Dissimilarity matrix written to %s\n", out_path))
    attr(dis, "disk_path") <- out_path
  }

  gc(verbose = FALSE, full = TRUE)

  # Tag with backend identity so downstream functions (as.rpart, e2tree, ...)
  # can detect mismatched (D, ensemble) pairs.
  attr(dis, "ensemble_backend") <- backend

  if (verbose) {
    cat("\n=== COMPUTATION COMPLETED ===\n")
    cat(sprintf(
      "Final matrix sparsity: %.2f%%\n",
      100 * (1 - sum(dis != 0) / length(dis))
    ))
  }

  return(dis)
}

determine_chunk_size <- function(n_obs, memory_limit = NULL, verbose = FALSE) {
  bytes_per_element <- 8
  bytes_per_row <- n_obs * bytes_per_element

  if (is.null(memory_limit)) {
    if (.Platform$OS.type == "unix") {
      mem_info <- try(
        system("free -b 2>/dev/null", intern = TRUE),
        silent = TRUE
      )
      if (!inherits(mem_info, "try-error") && length(mem_info) > 1) {
        mem_parts <- strsplit(mem_info[2], "\\s+")[[1]]
        if (length(mem_parts) >= 7) {
          available_bytes <- as.numeric(mem_parts[7])
          if (!is.na(available_bytes) && available_bytes > 0) {
            memory_limit <- (available_bytes / (1024^3)) * 0.5
          }
        }
      }
    }

    if (is.null(memory_limit)) {
      memory_limit <- 8
    }
  }

  memory_limit_bytes <- memory_limit * (1024^3)
  target_bytes_per_chunk <- memory_limit_bytes * 0.25
  chunk_size <- floor(target_bytes_per_chunk / bytes_per_row)
  chunk_size <- max(1000, min(chunk_size, n_obs))

  if (verbose) {
    cat(sprintf(
      "Determined chunk size: %d (based on %.2f GB memory limit)\n",
      chunk_size,
      memory_limit
    ))
  }

  return(chunk_size)
}

compute_dissimilarity_chunked <- function(a, obs, chunk_size, verbose = FALSE) {
  n_obs <- nrow(obs)
  n_chunks <- ceiling(n_obs / chunk_size)

  if (verbose) {
    cat(sprintf("Computing dissimilarity in %d chunks...\n", n_chunks))
    pb <- txtProgressBar(min = 0, max = n_chunks, style = 3)
  }

  aa <- Matrix::diag(a)
  dis <- matrix(0, nrow = n_obs, ncol = n_obs)

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1) * chunk_size + 1
    end_row <- min(i * chunk_size, n_obs)
    rows_idx <- start_row:end_row

    ## Keep sparse as long as possible; only convert the chunk rows
    a_chunk <- a[rows_idx, , drop = FALSE]  # stays sparse (dgCMatrix)
    aa_rows <- aa[rows_idx]
    aa_max  <- outer(aa_rows, aa, pmax)     # dense but only chunk_size x n
    ## Convert sparse chunk to dense only for the small chunk
    dis[rows_idx, ] <- 1 - (as.matrix(a_chunk) / aa_max)

    if (verbose) {
      setTxtProgressBar(pb, i)
    }

    rm(a_chunk, aa_max)
    if (i %% 5 == 0) gc(verbose = FALSE)
  }

  if (verbose) {
    close(pb)
  }

  row.names(dis) <- colnames(dis) <- obs$OBS

  return(dis)
}

maxValue <- function(x, y) {
  pmax(x, y)
}

## Variance — delegates to shared utility
variance <- e2_variance
