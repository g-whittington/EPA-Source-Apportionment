# Programmer: George Whittington
# Date: Purpose: Defied functions to help in data exploration

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
  data[[col_name]] <- lubridate::mdy(data[[col_name]])

  return(data)
}