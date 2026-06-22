# ============================================================================
# data-raw/panel_health.R
#
# Reproducible generator for `panel_health`: a SIMULATED country x year panel of
# life-expectancy determinants, shipped with e2tree to illustrate
# `panel_e2tree()`. The data-generating process is designed so that the two
# sources of variation are interpretable:
#
#   * BETWEEN drivers (cross-country level): development latent, GDP per capita,
#     health expenditure, schooling -- these mostly distinguish countries.
#   * WITHIN drivers (movement over time within a country): immunization gains,
#     undernourishment shocks, and a common medical-progress trend.
#
# NOTE: the data are SIMULATED (no real country is represented). Re-run this
# script to regenerate data/panel_health.rda.
#
#   source("data-raw/panel_health.R")
# ============================================================================

set.seed(20240608)

C      <- 30L                       # number of (synthetic) countries
years  <- 2004:2019                 # 16 years
Tt     <- length(years)
id     <- factor(sprintf("C%02d", rep(seq_len(C), each = Tt)))
year   <- rep(years, times = C)
yc     <- year - min(years)         # years since start (0..15)

# ---- country-level development latent (the engine of BETWEEN variation) -----
dev    <- rep(scale(rnorm(C))[, 1], each = Tt)          # ~ N(0,1) per country
cycle  <- as.numeric(stats::arima.sim(list(ar = 0.5), n = C * Tt, sd = 0.25))
shock  <- ifelse(runif(C * Tt) < 0.08, runif(C * Tt, 3, 9), 0)  # rare crises

# ---- observable features ----------------------------------------------------
gdp_pc <- round(exp(8.6 + 0.85 * dev + 0.020 * yc + cycle))                # USD
health_exp   <- round(pmin(14, pmax(2, 5.0 + 1.5 * dev + 0.035 * yc +
                                        rnorm(C * Tt, 0, 0.4))), 2)        # % GDP
schooling    <- round(pmin(15, pmax(3, 7.5 + 2.1 * dev + 0.055 * yc +
                                        rnorm(C * Tt, 0, 0.25))), 2)       # years

# immunization: logistic catch-up over time toward a country ceiling -> WITHIN
ceiling_i    <- 80 + 14 * stats::plogis(dev)
start_i      <- 55 + 12 * stats::plogis(dev)
immunization <- round(pmin(99, pmax(40,
                  ceiling_i - (ceiling_i - start_i) * exp(-0.18 * yc) +
                    rnorm(C * Tt, 0, 1.2))), 1)                            # %

sanitation   <- round(pmin(99, pmax(30, 58 + 16 * stats::plogis(dev) +
                                        0.5 * yc + rnorm(C * Tt, 0, 1.0))), 1)  # %

undernourish <- round(pmax(2, 18 - 7 * dev - 0.30 * yc + shock +
                              rnorm(C * Tt, 0, 0.6)), 1)                   # %

# ---- outcome: life expectancy (between level + within movement + noise) -----
life_expectancy <- round(
  56 +
    4.0 * dev +                       # BETWEEN: development level (high ICC)
    1.8 * log(gdp_pc / 1000) +        # BETWEEN (income), some within
    0.05 * immunization +             # WITHIN: immunization gains
    -0.15 * undernourish +            # WITHIN: undernourishment shocks
    0.12 * yc +                       # WITHIN: common medical-progress trend
    rnorm(C * Tt, 0, 0.5),
  2)

panel_health <- data.frame(
  country         = id,
  year            = year,
  gdp_pc          = gdp_pc,
  health_exp      = health_exp,
  schooling       = schooling,
  immunization    = immunization,
  sanitation      = sanitation,
  undernourish    = undernourish,
  life_expectancy = life_expectancy,
  stringsAsFactors = FALSE
)

# ---- quick provenance diagnostics (printed when sourced) --------------------
icc <- {
  mu  <- tapply(panel_health$life_expectancy, panel_health$country, mean)
  vb  <- stats::var(mu[as.character(panel_health$country)])
  vt  <- stats::var(panel_health$life_expectancy)
  vb / vt
}
message(sprintf("panel_health: %d countries x %d years = %d rows | LE in [%.1f, %.1f] | ICC = %.2f",
                C, Tt, nrow(panel_health),
                min(life_expectancy), max(life_expectancy), icc))

usethis_available <- requireNamespace("usethis", quietly = TRUE)
if (usethis_available) {
  usethis::use_data(panel_health, overwrite = TRUE)
} else {
  save(panel_health, file = "data/panel_health.rda", compress = "bzip2")
}
