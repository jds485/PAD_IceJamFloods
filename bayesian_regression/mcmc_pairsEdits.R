#Code source
#https://rdrr.io/cran/bayesplot/src/R/mcmc-scatterplots.R#sym-is_lower_tri
#Edits are made to the mcmc_pairs function to add xlim and ylim args.


check_ignored_arguments <- function(..., ok_args = character()) {
  dots <- list(...)
  if (length(dots)) {
    unrecognized <- if (!length(ok_args))
      names(dots) else setdiff(names(dots), ok_args)
    if (length(unrecognized)) {
      warn(paste(
        "The following arguments were unrecognized and ignored:",
        paste(unrecognized, collapse = ", ")
      ))
    }
  }
}

pairs_plotfun <- function(x) {
  fun <- paste0("mcmc_", x)
  utils::getFromNamespace(fun, "bayesplot")
}

prepare_mcmc_array <- function(x,
                               pars = character(),
                               regex_pars = character(),
                               transformations = list()) {
  if (is_df_with_chain(x)) {
    x <- df_with_chain2array(x)
  } else if (is_chain_list(x)) {
    # this will apply to mcmc.list and similar objects
    x <- chain_list2array(x)
  } else if (is.data.frame(x)) {
    # data frame without Chain column
    x <- as.matrix(x)
  } else {
    # try object's as.array method
    x <- as.array(x)
  }
  
  stopifnot(is.matrix(x) || is.array(x))
  if (is.array(x) && !(length(dim(x)) %in% c(2,3))) {
    abort("Arrays should have 2 or 3 dimensions. See help('MCMC-overview').")
  }
  if (anyNA(x)) {
    abort("NAs not allowed in 'x'.")
  }
  
  if (rlang::is_quosures(pars)) {
    pars <- tidyselect_parameters(complete_pars = parameter_names(x),
                                  pars_list = pars)
  } else {
    pars <- select_parameters(complete_pars = parameter_names(x),
                              explicit = pars,
                              patterns = regex_pars)
  }
  
  # possibly recycle transformations (apply same to all pars)
  if (is.function(transformations) ||
      (is.character(transformations) && length(transformations) == 1)) {
    transformations <- rep(list(transformations), length(pars))
    transformations <- set_names(transformations, pars)
  }
  
  if (is.matrix(x)) {
    x <- x[, pars, drop=FALSE]
    if (length(transformations)) {
      x <- apply_transformations(x, transformations)
    }
    x <- array(x, dim = c(nrow(x), 1, ncol(x)))
  } else {
    x <- x[, , pars, drop = FALSE]
    if (length(transformations)) {
      x <- apply_transformations(x, transformations)
    }
  }
  
  pars <- rename_transformed_pars(pars, transformations)
  set_mcmc_dimnames(x, pars)
}

is_df_with_chain <- function(x) {
  is.data.frame(x) && any(tolower(colnames(x)) %in% "chain")
}

# Convert list of matrices to 3-D array
chain_list2array <- function(x) {
  x <- validate_chain_list(x)
  n_chain <- length(x)
  if (n_chain == 1) {
    n_iter <- nrow(x[[1]])
    param_names <- colnames(x[[1]])
  } else {
    n_iter <- sapply(x, nrow)
    cnames <- sapply(x, colnames)
    param_names <- if (is.array(cnames))
      cnames[, 1] else cnames
    n_iter <- n_iter[1]
  }
  param_names <- unique(param_names)
  n_param <- length(param_names)
  out <- array(NA, dim = c(n_iter, n_chain, n_param))
  for (i in seq_len(n_chain)) {
    out[, i,] <- x[[i]]
  }
  
  set_mcmc_dimnames(out, param_names)
}

is_chain_list <- function(x) {
  check1 <- !is.data.frame(x) && is.list(x)
  dims <- try(sapply(x, function(chain) length(dim(chain))), silent=TRUE)
  if (inherits(dims, "try-error")) {
    return(FALSE)
  }
  check2 <- isTRUE(all(dims == 2)) # all elements of list should be matrices/2-D arrays
  check1 && check2
}

validate_chain_list <- function(x) {
  n_chain <- length(x)
  for (i in seq_len(n_chain)) {
    nms <- colnames(as.matrix(x[[i]]))
    if (is.null(nms) || !all(nzchar(nms))) {
      abort(paste(
        "Some parameters are missing names.",
        "Check the column names for the matrices in your list of chains."
      ))
    }
  }
  if (n_chain > 1) {
    n_iter <- sapply(x, nrow)
    same_iters <- length(unique(n_iter)) == 1
    if (!same_iters) {
      abort("Each chain should have the same number of iterations.")
    }
    
    cnames <- sapply(x, colnames)
    if (is.array(cnames)) {
      same_params <- identical(cnames[, 1], cnames[, 2])
    } else {
      same_params <- length(unique(cnames)) == 1
    }
    if (!same_params) {
      abort(paste(
        "The parameters for each chain should be in the same order",
        "and have the same names."
      ))
    }
  }
  
  x
}

is_3d_array <- function(x) {
  if (!is.array(x)) {
    return(FALSE)
  }
  
  if (length(dim(x)) != 3) {
    return(FALSE)
  }
  
  TRUE
}

set_mcmc_dimnames <- function(x, parnames) {
  stopifnot(is_3d_array(x))
  dimnames(x) <- list(
    Iteration = seq_len(nrow(x)),
    Chain = seq_len(ncol(x)),
    Parameter = parnames
  )
  structure(x, class = c(class(x), "mcmc_array"))
}

select_parameters <-
  function(explicit = character(),
           patterns = character(),
           complete_pars = character()) {
    
    stopifnot(is.character(explicit),
              is.character(patterns),
              is.character(complete_pars))
    
    if (!length(explicit) && !length(patterns)) {
      return(complete_pars)
    }
    
    if (length(explicit)) {
      if (!all(explicit %in% complete_pars)) {
        not_found <- which(!explicit %in% complete_pars)
        abort(paste(
          "Some 'pars' don't match parameter names:",
          paste(explicit[not_found], collapse = ", "),
          call. = FALSE
        ))
      }
    }
    
    if (!length(patterns)) {
      return(unique(explicit))
    } else {
      regex_pars <-
        unlist(lapply(seq_along(patterns), function(j) {
          grep(patterns[j], complete_pars, value = TRUE)
        }))
      
      if (!length(regex_pars)) {
        abort("No matches for 'regex_pars'.")
      }
    }
    
    unique(c(explicit, regex_pars))
}

# Get parameter names from a 3-D array
parameter_names <- function(x) UseMethod("parameter_names")
parameter_names.array <- function(x) {
  stopifnot(is_3d_array(x))
  dimnames(x)[[3]] %||% abort("No parameter names found.")
}
parameter_names.default <- function(x) {
  colnames(x) %||% abort("No parameter names found.")
}
parameter_names.matrix <- function(x) {
  colnames(x) %||% abort("No parameter names found.")
}

rename_transformed_pars <- function(pars, transformations) {
  stopifnot(is.character(pars), is.list(transformations))
  has_names <- sapply(transformations, is.character)
  if (any(has_names)) {
    nms <- names(which(has_names))
    for (nm in nms) {
      pars[which(pars == nm)] <-
        paste0(
          transformations[[nm]], "(",
          pars[which(pars == nm)], ")"
        )
    }
  }
  if (any(!has_names)) {
    nms <- names(which(!has_names))
    pars[pars %in% nms] <- paste0("t(", pars[pars %in% nms], ")")
  }
  
  pars
}

#' Drop any constant or duplicate variables
#' @noRd
#' @param x 3-D array
drop_constants_and_duplicates <- function(x) {
  x2 <- drop_consts(x)
  x2 <- drop_dupes(x2)
  class(x2) <- c(class(x2), "mcmc_array")
  x2
}
drop_consts <- function(x) {
  varying <- apply(x, 3, FUN = function(y) length(unique(c(y))) > 1)
  if (all(varying))
    return(x)
  
  warn(paste(
    "The following parameters were dropped because they are constant:",
    paste(names(varying)[!varying], collapse = ", ")
  ))
  x[, , varying, drop = FALSE]
}
drop_dupes <- function(x) {
  dupes <- duplicated(x, MARGIN = 3)
  if (!any(dupes))
    return(x)
  
  warn(paste(
    "The following parameters were dropped because they are duplicative:",
    paste(parameter_names(x)[dupes], collapse = ", ")
  ))
  x[, , !dupes, drop = FALSE]
}

num_chains <- function(x, ...) UseMethod("num_chains")
num_iters <- function(x, ...) UseMethod("num_iters")
num_params <- function(x, ...) UseMethod("num_params")
num_params.mcmc_array <- function(x, ...) dim(x)[3]
num_chains.mcmc_array <- function(x, ...) dim(x)[2]
num_iters.mcmc_array <- function(x, ...) dim(x)[1]
num_params.data.frame <- function(x, ...) {
  stopifnot("Parameter" %in% colnames(x))
  length(unique(x$Parameter))
}
num_chains.data.frame <- function(x, ...) {
  stopifnot("Chain" %in% colnames(x))
  length(unique(x$Chain))
}
num_iters.data.frame <- function(x, ...) {
  cols <- colnames(x)
  stopifnot("Iteration" %in% cols || "Draws" %in% cols)
  
  if ("Iteration" %in% cols) {
    n <- length(unique(x$Iteration))
  } else {
    n <- length(unique(x$Draw))
  }
  
  n
}

#' Handle user's specified `condition`
#' @noRd
#' @param x 3-D mcmc array.
#' @param condition Object returned by `pairs_condition()`.
#' @param np,lp User-specified arguments to `mcmc_pairs()`.
#' @return A named list containing `"x"` (`x`, possibly modified) and `"mark"`
#'   (logical or interger vector for eventually splitting `x`).
handle_condition <- function(x, condition=NULL, np=NULL, lp=NULL) {
  n_iter <- num_iters(x)
  n_chain <- num_chains(x)
  no_np <- is.null(np)
  no_lp <- is.null(lp)
  
  cond_type <- attr(condition, "type")
  
  if (cond_type == "default") {
    k <- ncol(x) %/% 2
    mark <- c(rep(FALSE, n_iter * k), rep(TRUE, n_iter * (n_chain - k)))
  } else if (cond_type == "chain_vector") {
    x <- x[, condition, , drop = FALSE]
    k <- ncol(x) %/% 2
    n_chain <- length(condition)
    mark <- c(rep(FALSE, n_iter * k), rep(TRUE, n_iter * (n_chain - k)))
  } else if (cond_type == "chain_list") {
    x <- x[, c(condition[[1]], condition[[2]]), , drop = FALSE]
    k1 <- length(condition[[1]])
    k2 <- length(condition[[2]])
    mark <- c(rep(TRUE, n_iter * k1), rep(FALSE, n_iter * k2))
  } else if (cond_type == "draws_proportion") {
    mark <- rep(1:n_iter > (condition * n_iter), times = n_chain)
  } else if (cond_type == "draws_selection") {
    # T/F for each iteration to split into upper and lower
    stopifnot(length(condition) == (n_iter * n_chain))
    mark <- !condition
  } else if (cond_type == "nuts") {
    # NUTS sampler param or lp__
    if (no_np && condition != "lp__")
      abort(paste(
        "To use this value of 'condition' the 'np' argument",
        "to 'mcmc_pairs' must also be specified."
      ))
    
    if (condition == "lp__") {
      if (no_lp)
        abort(paste(
          "If 'condition' is 'lp__' then the 'lp' argument",
          "to 'mcmc_pairs' must also be specified."
        ))
      mark <- unstack_to_matrix(lp, Value ~ Chain)
      
    } else {
      param <- sym("Parameter")
      mark <- dplyr::filter(np, UQ(param) == condition)
      mark <- unstack_to_matrix(mark, Value ~ Chain)
    }
    if (condition == "divergent__") {
      mark <- as.logical(mark)
    } else {
      mark <- c(mark) >= median(mark)
    }
    if (length(unique(mark)) == 1)
      abort(paste(condition, "is constant so it cannot be used as a condition."))
  }
  
  list(x = x, mark = mark)
}

#' Convert 3-D array to matrix with chains merged
#'
#' @noRd
#' @param x A 3-D array (iter x chain x param)
#' @return A matrix with one column per parameter
#'
merge_chains <- function(x) {
  xdim <- dim(x)
  mat <- array(x, dim = c(prod(xdim[1:2]), xdim[3]))
  colnames(mat) <- parameter_names(x)
  mat
}

#' Check if off-diagonal plot is above or below the diagonal
#'
#' @noRd
#' @param j integer (index)
#' @param n Number of parameters (number of plots = `n^2`)
#' @return `TRUE` if below the diagonal, `FALSE` if above the diagonal
is_lower_tri <- function(j, n) {
  idx <- array_idx_j(j, n)
  lower_tri <- lower_tri_idx(n)
  row_match_found(idx, lower_tri)
}
#' Get array indices of the jth element in the plot matrix
#'
#' @noRd
#' @param j integer (index)
#' @param n number of parameters (number of plots = n^2)
#' @return rwo vector (1-row matrix) containing the array indices of the jth
#'   element in the plot matrix
array_idx_j <- function(j, n) {
  jj <- matrix(seq_len(n^2), nrow = n, byrow = TRUE)[j]
  arrayInd(jj, .dim = c(n, n))
}

#' Get indices of lower triangular elements of a square matrix
#' @noRd
#' @param n number of rows (columns) in the square matrix
lower_tri_idx <- function(n) {
  a <- rev(abs(sequence(seq.int(n - 1)) - n) + 1)
  b <- rep.int(seq.int(n - 1), rev(seq.int(n - 1)))
  cbind(row = a, col = b)
}

#' Find which (if any) row in y is a match for x
#' @noRd
#' @param x a row vector (i.e., a matrix with 1 row)
#' @param y a matrix
#' @return either a row number in `y` or `NA` if no match
row_match_found <- function(x, y) {
  stopifnot(is.matrix(x), is.matrix(y), nrow(x) == 1)
  x <- as.data.frame(x)
  y <- as.data.frame(y)
  res <- match(
    do.call(function(...) paste(..., sep=":::"), x),
    do.call(function(...) paste(..., sep=":::"), y)
  )
  isTRUE(!is.na(res) && length(res) == 1)
}


mcmc_pairs = function (x, pars = character(), regex_pars = character(), transformations = list(), 
          ..., diag_fun = c("hist", "dens"), off_diag_fun = c("scatter", 
                                                              "hex"), diag_args = list(), off_diag_args = list(), 
          condition = pairs_condition(), lp = NULL, np = NULL, np_style = pairs_style_np(), 
          max_treedepth = NULL, grid_args = list(), save_gg_objects = TRUE, xlim=NULL, ylim=NULL) 
{
  check_ignored_arguments(...)
  stopifnot(is.list(diag_args), is.list(off_diag_args), inherits(np_style, 
                                                                 "nuts_style"), inherits(condition, "pairs_condition"))
  plot_diagonal <- pairs_plotfun(match.arg(diag_fun))
  plot_off_diagonal <- pairs_plotfun(match.arg(off_diag_fun))
  x <- prepare_mcmc_array(x, pars, regex_pars, transformations)
  x <- drop_constants_and_duplicates(x)
  n_iter <- num_iters(x)
  n_chain <- num_chains(x)
  n_param <- num_params(x)
  pars <- parameter_names(x)
  if (n_chain == 1) {
    warning("Only one chain in 'x'. This plot is more useful with multiple chains.")
  }
  if (n_param < 2) {
    stop("This plot requires at least two parameters in 'x'.")
  }
  no_np <- is.null(np)
  no_lp <- is.null(lp)
  no_max_td <- is.null(max_treedepth)
  if (!no_np) {
    param <- sym("Parameter")
    val <- sym("Value")
    np <- validate_nuts_data_frame(np, lp)
    divs <- dplyr::filter(np, UQ(param) == "divergent__") %>% 
      pull(UQ(val))
    divergent__ <- matrix(divs, nrow = n_iter * n_chain, 
                          ncol = n_param)[, 1]
    if (!no_max_td) {
      gt_max_td <- (dplyr::filter(np, UQ(param) == "treedepth__") %>% 
                      pull(UQ(val))) > max_treedepth
      max_td_hit__ <- matrix(gt_max_td, nrow = n_iter * 
                               n_chain, ncol = n_param)[, 1]
    }
  }
  cond <- handle_condition(x, condition, np, lp)
  x <- merge_chains(cond[["x"]])
  mark <- cond[["mark"]]
  all_pairs <- expand.grid(pars, pars, stringsAsFactors = FALSE, 
                           KEEP.OUT.ATTRS = FALSE)
  plots <- vector("list", length = nrow(all_pairs))
  use_default_binwidth <- is.null(diag_args[["binwidth"]])
  for (j in seq_len(nrow(all_pairs))) {
    pair <- as.character(all_pairs[j, ])
    if (identical(pair[1], pair[2])) {
      diag_args[["x"]] <- x[, pair[1], drop = FALSE]
      if (diag_fun == "hist" && use_default_binwidth) 
        diag_args[["binwidth"]] <- diff(range(diag_args[["x"]]))/30
      plots[[j]] <- do.call(plot_diagonal, diag_args) + 
        labs(subtitle = pair[1]) + theme(axis.line.y = element_blank(), 
                                         plot.subtitle = element_text(hjust = 0.5))
    }
    else {
      mark2 <- if (is_lower_tri(j, n_param)) 
        !mark
      else mark
      x_j <- x[mark2, pair, drop = FALSE]
      if (!no_np) {
        divs_j <- divergent__[mark2]
        max_td_hit_j <- if (no_max_td) 
          NULL
        else max_td_hit__[mark2]
      }
      else {
        divs_j <- max_td_hit_j <- NULL
      }
      off_diag_args[["x"]] <- x_j
      plots[[j]] <- do.call(plot_off_diagonal, off_diag_args)
      if (isTRUE(any(divs_j == 1))) {
        divs_j_fac <- factor(as.logical(divs_j), levels = c(FALSE, 
                                                            TRUE), labels = c("NoDiv", "Div"))
        plots[[j]] <- plots[[j]] + geom_point(aes_(color = divs_j_fac, 
                                                   size = divs_j_fac), shape = np_style$shape[["div"]], 
                                              alpha = np_style$alpha[["div"]], na.rm = TRUE)
      }
      if (isTRUE(any(max_td_hit_j == 1))) {
        max_td_hit_j_fac <- factor(max_td_hit_j, levels = c(FALSE, 
                                                            TRUE), labels = c("NoHit", "Hit"))
        plots[[j]] <- plots[[j]] + geom_point(aes_(color = max_td_hit_j_fac, 
                                                   size = max_td_hit_j_fac), shape = np_style$shape[["td"]], 
                                              alpha = np_style$alpha[["td"]], na.rm = TRUE)
      }
      if (isTRUE(any(divs_j == 1)) || isTRUE(any(max_td_hit_j == 
                                                 1))) 
        plots[[j]] <- format_nuts_points(plots[[j]], 
                                         np_style)
    }
    if (!is.null(xlim) & !is.null(ylim)){
      plots[[j]] <- plots[[j]] + coord_cartesian(xlim = xlim[j,], ylim = ylim[j,])
    }else if (!is.null(xlim)){
      plots[[j]] <- plots[[j]] + coord_cartesian(xlim = xlim[j,])
    }else if (!is.null(ylim)){
      plots[[j]] <- plots[[j]] + coord_cartesian(ylim = ylim[j,])
    }
  }
  plots <- lapply(plots, function(x) x + xaxis_title(FALSE) + 
                    yaxis_title(FALSE))
  bayesplot_grid(plots = plots, legends = FALSE, grid_args = grid_args, 
                 save_gg_objects = save_gg_objects)
}
