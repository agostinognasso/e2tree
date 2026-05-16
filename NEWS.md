# e2tree (development version)

# e2tree 1.2.0

## Robustness fixes (2026-05)

- **CatBoost multi-class leaf extraction**: for multi-class
  objectives (`MultiClass`, `MultiClassOneVsAll`)
  `catboost.predict(..., prediction_type = "RawFormulaVal", ntree_start,
  ntree_end)` returns an `n_obs x n_classes` matrix rather than a
  scalar vector, which broke the leaf-proxy assignment with
  `Error in leaf_mat[, t] <- as.integer(factor(round(raw, digits = 10L))) :
  number of items to replace is not a multiple of replacement length`.
  The CatBoost adapter now collapses the per-tree raw-score matrix into a
  single key per observation (concatenation of the rounded class scores)
  before discretising into a leaf proxy. The single-output regression
  case is unchanged.
- **CatBoost loss-function parsing**: in some catboost releases the
  `loss_function` returned by `catboost.get_model_params()` is a
  list (e.g. `list(type = "MultiClass", border = 0.5)`) instead of a
  bare character scalar. The previous detection code piped this
  list directly into `startsWith()` and failed with
  `Error: non-character object(s)`. The new
  `.catboost_extract_loss()` canonicalises any of the documented
  shapes (plain character, list-with-`type`, list-with-`name`, list
  whose first element holds the name) to a length-one character
  string before classification/regression dispatch and falls back to
  `attr(ensemble, "params")` when `catboost.get_model_params()` does
  not expose the loss.
- **CatBoost 1.2.x compatibility**: starting from `catboost` 1.2 the
  R objects returned by `catboost.train()` carry the class
  `catboost.Model` instead of the previous `catboost.CatBoost`.
  All three S3 generics in the adapter layer (`get_ensemble_type`,
  `extract_terminal_nodes`, `get_ensemble_predictions`) now register
  methods for both class names, sharing a single internal worker, so
  models trained with either version are accepted out of the box.
  `ensemble_backend()` and `.SUPPORTED_BACKENDS` recognise both class
  names. `createDisMatrix()` now sets the `e2tree_label` attribute on
  CatBoost ensembles automatically when the user passes `label`,
  eliminating the previously required manual
  `attr(ensemble, "e2tree_label") <- ...` step.
- **`moda()` accepts non-factor responses**: classification ensembles
  trained on integer 0/1 responses (typical of `gbm` with the
  `bernoulli` distribution) no longer trigger
  `Error in res[1] <- as.character(levels(x)[res[1]]) : replacement
  has length zero`. The internal `moda()` helper now coerces the
  modal value to character independently of whether the response is
  a factor, an integer, a character or a numeric vector. Affected
  call sites: `eStoppingRules()`, `e2tree()`, `vimp()`.
- **Vignette `models.Rmd` corrected**: the regression examples now
  pass the response column through `data` together with
  `label = "<response_name>"`, as required by `createDisMatrix()` for
  regression backends; the `gbm` regression example sets
  `n.minobsinnode = 2` and `bag.fraction = 0.8` so it does not fail
  the `gbm` internal sample-size check on the small `mtcars` training
  set; the `lightgbm` regression example uses `num_leaves = 8` and
  `min_data_in_leaf = 2`, which prevents the small-sample degenerate
  case caught by `validate_terminal_nodes()`. The classification
  example based on the binary `is_setosa` indicator now passes a
  factor copy of the response to `e2tree()` while leaving the
  numeric copy to `gbm`.

## Multi-backend hardening (2026-04)

- **Adapter contract validation**: `createDisMatrix()` now calls
  `validate_terminal_nodes()` immediately after `extract_terminal_nodes()`.
  Malformed adapter output (wrong row count, non-numeric leaf columns, or
  the all-same-leaf degenerate case that previously made GBM silently
  produce an all-zero `D`) is now rejected with an informative error.
- **Backend tagging**: dissimilarity matrices returned by `createDisMatrix()`
  carry a new `ensemble_backend` attribute identifying the backend used.
  `e2tree()` and `as.rpart()` warn when the supplied ensemble does not match
  this tag, catching the common "stale `D` reused with a different model"
  mistake.
- **xgboost 3.x compatibility**: models built via the high-level
  `xgboost::xgboost()` wrapper (which adds an `xgboost` class on top of
  `xgb.Booster`) are now handled correctly; previously the wrapper's
  `predict.xgboost` method rejected `xgb.DMatrix` input and broke leaf
  extraction.
- **Chunked dissimilarity path wired in**: when `chunk_size < n_obs` the
  function now actually uses `compute_dissimilarity_chunked()` (previously
  the parameter was accepted but ignored). Optional `use_disk = TRUE`
  persists the result via `saveRDS()`.
- **C++ type checks**: `compute_cooccurrences_cpp()`,
  `compute_cooccurrences_sparse_cpp()` and `compute_all_cooccurrences_cpp()`
  now validate that `Tree*` and `resp` columns are integer/double, raising
  a clear error rather than silently coercing factors/characters to zeros.
- **Adapter de-duplication**: the three S3 generics share a single
  `.unsupported_backend()` helper and a single supported-classes list;
  the xgboost adapters share a `.xgb_dmatrix()` builder.
- **Documentation**: `?createDisMatrix` now contains an "Interpretation
  note (RF vs boosting)" section explaining why the dissimilarity scale
  differs systematically between bagging and boosting backends.
- **Tests**: new `tests/testthat/test-multi-backend.R` exercises the D
  invariants (square, zero diagonal, symmetric, non-degenerate, correctly
  tagged) for each available backend, and covers the validator and the
  default-method failure paths.

## S3 class system overhaul

- `e2tree` class now properly listed as first in the class vector (`c("e2tree", "list")`).
- New S3 methods for `e2tree`: `predict()`, `fitted()`, `residuals()`.
- `predict.e2tree()` replaces `ePredTree()` as the standard prediction interface.
  For regression, returns a data frame with `fit` and `sd` (node-level standard deviation).
  For classification, returns a data frame with `fit`, `accuracy`, and `score`.
- `fitted.e2tree()` returns fitted values for training data.
- `residuals.e2tree()` returns residuals for regression E2Trees.
- `methods(class = "e2tree")` now shows: `as.rpart`, `e2splits`, `fitted`, `nodes`, `plot`, `predict`, `print`, `residuals`, `summary`.
- `methods(class = "eValidation")` now shows: `measures`, `plot`, `print`, `proximity`, `summary`.

## Accessor functions (new)

- `nodes()`: Extract tree node data frame from an `e2tree` object, with optional `terminal` filter.
- `e2splits()`: Extract split and categorical split information.
- `measures()`: Extract validation measures from an `eValidation` object.
- `proximity()`: Extract proximity matrices (ensemble, e2tree, or both) from an `eValidation` object.

## Coercion methods (new)

- `as.rpart()`: Generic and method for converting `e2tree` to `rpart` format.
- `as.party()`: Method for converting `e2tree` to `partykit`'s `constparty` format (registered conditionally when partykit is installed). Produces proper bar plots in terminal nodes for classification trees.
- `rpart2Tree()` retained for backward compatibility with a deprecation note.

## Validation framework improvements

- `eValidation()` gains a `test` argument: `"mantel"` (Mantel test only), `"measures"` (divergence/similarity measures only), or `"both"` (default). This allows choosing between association testing and agreement testing.
- `print.eValidation()`, `summary.eValidation()`, and `plot.eValidation()` updated to handle all three test modes gracefully.

## Variable importance improvements

- `vimp()`: Auto-detects classification/regression from the `e2tree` object; `type` argument now optional.
- Fixed incorrect y-axis label ("Variance" instead of "Variable") in variable importance plots.
- Regression variable importance bars now sorted by importance (previously unsorted).
- Consistent column naming (`Variable`, `MeanImpurityDecrease`) across classification and regression.
- Internal logic refactored into `.vimp_classification()` and `.vimp_regression()`.

## Documentation

- All man page titles standardized to Title Case.
- `\dontrun{}` replaced with `\donttest{}` in all examples; interactive-only examples wrapped in `if (interactive())`.
- Examples updated to use accessor functions and `predict()` instead of direct `$` access.
- New vignette `e2tree-introduction` covering classification, regression, validation (Mantel test, divergence measures, LoI decomposition), and comparison with partykit/stablelearner.
- `ePredTree()` documentation updated with deprecation note pointing to `predict.e2tree()`.

## Package infrastructure

- Added `partykit`, `knitr`, `rmarkdown` to Suggests.
- Added `VignetteBuilder: knitr` to DESCRIPTION.
- `e2tree` object now stores `fitted.values`, `y`, and `data` for S3 method support.
- Conditional `.onLoad` hook for registering `as.party.e2tree` when partykit is loaded.

# e2tree 1.0.0

## New functions

- `goi()`: Goodness of Interpretability (GoI) index measuring how well the E2Tree-estimated proximity matrix reconstructs the original ensemble proximity matrix.
- `goi_perm()`: Permutation test for the GoI index to assess statistical significance.
- `goi_analysis()`: Combined GoI analysis returning both the observed statistic and permutation results.
- `plot.goi_perm()`: Plot method for `goi_perm` objects displaying the permutation distribution.
- `plot_e2tree_vis()`: Interactive E2Tree visualization using `visNetwork` with draggable nodes, zoom/pan, and multiple layout options.
- `plot_e2tree_click()`: Interactive E2Tree plot in the R graphics device with click-to-inspect node details.
- `save_e2tree_html()`: Save an interactive `visNetwork` tree plot as a standalone HTML file.
- `print_e2tree_summary()`: Print a formatted summary of an e2tree object.

## Performance improvements

- `createDisMatrix`: C++ backend (`CoOccurrences.cpp`) with OpenMP thread-level parallelism replaces the R-level `foreach`/`doParallel` loop; co-occurrence normalization also moved to C++.
- `e2tree`: vectorized `Wt` computation using `vapply`; simplified internal `get_classes` helper.
- `ePredTree`: split rules are now pre-parsed once (`parse_all_splits` + `apply_split_rule`), eliminating repeated `regex`/`eval(parse())` calls during prediction.
- `vimp`: three `group_by` operations consolidated into one; `eval`/`parse` removed.
- `split`: `ordSplit` and `catSplit` vectorized with `outer()`.
- `eImpurity`: single-step integer matrix conversion.

## Bug fixes

- Fixed Rcpp type conversion (`NumericVector` + `static_cast<int>`) for ranger compatibility (ranger returns double columns for tree node assignments).
- Fixed `ePredTree` returning character instead of double for regression trees.
- Fixed bare `filter()` call in `proximity_longer.R` causing namespace conflict with `dplyr`.
- Added missing `NAMESPACE` imports for `graphics`, `grDevices`, and `utils`.

## Internal changes

- New `aaa_utils.R` with shared internal helpers: `e2_variance()`, `get_ensemble_type()`, `check_package()`.
- Refactored `eValidation`, `eImpurity`, and `eStoppingRules` for consistency and performance.

# e2tree 0.2.0

- Added support for 'ranger' models
- Several improvements in e2tree plots

# e2tree 0.1.2

# e2tree 0.1.1

# e2tree 0.0.0.9000

* Added a `NEWS.md` file to track changes to the package.
