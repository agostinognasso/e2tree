eStoppingRules <- function(y,index,t, setting, response, ensemble, vart1){
  n <- length(index)
  
  # create type object
  type <- get_ensemble_type(ensemble)
  
  if (n>1){
    impTotal <- meanDis(y[index,index])
    switch(type,
           classification = {
             res <- as.numeric(moda(response[index])[2])
           },
           regression={
             res <- 1-(variance(response[index])/vart1)
           }
    )
  } else {impTotal <- 0
  res <- 1
  }
  
  
  sRule <- isTRUE(impTotal<=setting$impTotal |
                    n<=setting$n |
                    res > 0.95  | ### if the variance is less than 5% of the variance in the root node, stop
                    (t*2)+1 > setting$tMax)
  results <- list(sRule=sRule,impTotal=impTotal,n=n)
  return(results)
  
}

############

meanDis <- function(dis){
  n <- nrow(dis)
  sum(dis)/(n*(n-1))
}

############

moda <- function(x) {
  if ( anyNA(x) ) x = x[!is.na(x)]
  ux <- unique(x)
  tab <- tabulate(match(x, ux))
  mode_val <- ux[which.max(tab)[1]]
  freq     <- max(tab)[1] / sum(tab)
  # `mode_val` may be a factor element, an integer (e.g., 0/1 binary
  # classification fed by gbm bernoulli), a character or a numeric value;
  # coerce uniformly to character so the caller can store it next to a
  # numeric frequency without triggering a length-zero replacement.
  c(as.character(mode_val), freq)
}