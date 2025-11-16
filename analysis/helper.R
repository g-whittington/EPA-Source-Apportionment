# Programmer: George Whittington
# Date: Purpose: Defied functions to help in data exploration

i_am("analysis/helper.R")

#' Fix EPA Date Formatting
#'
#' Converts a column containing "mm/dd/yy" strings into proper R Date objects.
#' Uses lubridate::mdy() for the conversion.
#'
#' @param data A tibble or dataframe containing the raw data.
#' @param col_name String. The name of the column to fix. Defaults to "Date".
#'
#' @return A tibble or dataframe with the specified column transformed to Date format.
#' @export
convert_to_date <- function(data, col_name = "Date") {
  # the Date columns are in different formats across datasets
  data |> 
    mutate(
      Date = parse_date_time(Date, orders = c("mdy", "mdy HM"))
    )
}

#' Convert EPA Data to Long Format with Uncertainty and Write to CSV
#' 
#' Takes both a concentration and uncertainty file and pivots them to a long format
#' and combines then into one csv to be further analyzed 
#' 
#' @param data_con A tibble or dataframe containing the concentration data
#' @param data_unc A tibble or dataframe containing the uncertainty data
#' @param file_name String. file name to save data as
#' 
#' @return NUll. It creates a combined csv file
#' @export
write_long_data <- function(data_con, data_unc, file_name) {
  # keep the data column as it and move the names and values of each component to a row
  data_con_long <- data_con |> 
    pivot_longer(
      -Date,
      names_to = "Component",
      values_to = "Concentration"
    )
  
  # keep the data column as it and move the names and values of each component to a row
  data_unc_long <- data_unc |> 
    pivot_longer(
      -Date,
      names_to = "Component",
      values_to = "Uncertainty"
    )
  
  # join them together and add uncertainty percentage
  combined <- data_con_long |> 
    full_join(
      data_unc_long,
      join_by(Date, Component)
    ) |> 
      mutate(
        Uncertainty_Ratio = Uncertainty / Concentration
      )
  
  write_csv(combined, here("data", file_name))
}
