---
title: "Predictive Classification of In-Vehicle Coupon Recommendation"
author: "Omer Mejia"
date: "9/4/2021"
output: 
  html_document:
    keep_md: true
---


  
## A. Project Goals and Approach

### 1. Project Goal and Significance

The goal is to build a preliminary classification model to predict whether the target person will accept a coupon offer given the conditions.

I have particular interest in this type of data (marketing) because it is highly relevant to other promotion related endeavors in the non-profit sector such as fundraising and influencing people to perform positive behaviors.


### 3. Approach 

This study will explore multiple supervised learning algorithms and determine  which among the models developed best predicts the behavior or choice of the person.

To prepare and conduct the classification prediction model, this project will utilize the `tidverse` and `tidymodels` packages in R. 

The `tidymodels` package provides a systematic and unified interface in conducting machine learning models using various algorithms. The machine learning algorithms that will be used will be discussed later in the modeling section.




### 4. Load packages

This project will use the `tidyverse` and `tidymodels` packages.


```r
library(tidyverse) # data analysis and modeling
```

```
## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --
```

```
## v ggplot2 3.3.5     v purrr   0.3.4
## v tibble  3.1.2     v dplyr   1.0.7
## v tidyr   1.1.3     v stringr 1.4.0
## v readr   1.4.0     v forcats 0.5.1
```

```
## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```r
library(tidymodels) # data analysis and modeling
```

```
## Registered S3 method overwritten by 'tune':
##   method                   from   
##   required_pkgs.model_spec parsnip
```

```
## -- Attaching packages -------------------------------------- tidymodels 0.1.3 --
```

```
## v broom        0.7.8      v rsample      0.1.0 
## v dials        0.0.9      v tune         0.1.5 
## v infer        0.5.4      v workflows    0.2.2 
## v modeldata    0.1.0      v workflowsets 0.0.2 
## v parsnip      0.1.6      v yardstick    0.0.8 
## v recipes      0.1.16
```

```
## -- Conflicts ----------------------------------------- tidymodels_conflicts() --
## x scales::discard() masks purrr::discard()
## x dplyr::filter()   masks stats::filter()
## x recipes::fixed()  masks stringr::fixed()
## x dplyr::lag()      masks stats::lag()
## x yardstick::spec() masks readr::spec()
## x recipes::step()   masks stats::step()
## * Use tidymodels_prefer() to resolve common conflicts.
```

```r
library(naniar) # analysis of missing data
library(DataExplorer) # analysis of missing data
library(visdat) # analysis of missing data
```

## B. Data Exploration and Cleaning

### 1. Dataset Information

The dataset was downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation). It was uploaded on the website on September 15, 2020.

Tong Wang (University of Iowa) and Cynthia Rudin (Duke University) published a paper in 2017 that used this data. They shared the data to UCI ML Repository.  

From the dataset information in the UCI: "This data was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver."

Data Set Characteristics:  Multivariate
Number of Instances: 12684
Area: Business
Attribute Characteristics: N/A
Number of Attributes: 23
Date Donated: 2020-09-15
Missing Values? Yes



### 2. Data Ingestion 

The file is in a CSV format. The data will be imported as `accept_coupon` using `read.csv`


```r
accept_coupon <- read.csv("in-vehicle-coupon-recommendation.csv")
```


```r
str(accept_coupon)
```

```
## 'data.frame':	12684 obs. of  26 variables:
##  $ destination         : chr  "No Urgent Place" "No Urgent Place" "No Urgent Place" "No Urgent Place" ...
##  $ passanger           : chr  "Alone" "Friend(s)" "Friend(s)" "Friend(s)" ...
##  $ weather             : chr  "Sunny" "Sunny" "Sunny" "Sunny" ...
##  $ temperature         : int  55 80 80 80 80 80 55 80 80 80 ...
##  $ time                : chr  "2PM" "10AM" "10AM" "2PM" ...
##  $ coupon              : chr  "Restaurant(<20)" "Coffee House" "Carry out & Take away" "Coffee House" ...
##  $ expiration          : chr  "1d" "2h" "2h" "2h" ...
##  $ gender              : chr  "Female" "Female" "Female" "Female" ...
##  $ age                 : chr  "21" "21" "21" "21" ...
##  $ maritalStatus       : chr  "Unmarried partner" "Unmarried partner" "Unmarried partner" "Unmarried partner" ...
##  $ has_children        : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ education           : chr  "Some college - no degree" "Some college - no degree" "Some college - no degree" "Some college - no degree" ...
##  $ occupation          : chr  "Unemployed" "Unemployed" "Unemployed" "Unemployed" ...
##  $ income              : chr  "$37500 - $49999" "$37500 - $49999" "$37500 - $49999" "$37500 - $49999" ...
##  $ car                 : chr  "" "" "" "" ...
##  $ Bar                 : chr  "never" "never" "never" "never" ...
##  $ CoffeeHouse         : chr  "never" "never" "never" "never" ...
##  $ CarryAway           : chr  "" "" "" "" ...
##  $ RestaurantLessThan20: chr  "4~8" "4~8" "4~8" "4~8" ...
##  $ Restaurant20To50    : chr  "1~3" "1~3" "1~3" "1~3" ...
##  $ toCoupon_GEQ5min    : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ toCoupon_GEQ15min   : int  0 0 1 1 1 1 1 1 1 1 ...
##  $ toCoupon_GEQ25min   : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ direction_same      : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ direction_opp       : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ Y                   : int  1 0 1 0 0 1 1 1 1 0 ...
```

	
The attribute information can be found in the dataset webpage <https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation>

The outcome variable is `Y` which indicates `1` (accepted the coupon) and `0` (not accepted the coupon)


### 3. Data Exploration

First, we will explore the data using the built in R `glimpse()` and `summary()` functions. 


```r
glimpse(accept_coupon)
```

```
## Rows: 12,684
## Columns: 26
## $ destination          <chr> "No Urgent Place", "No Urgent Place", "No Urgent ~
## $ passanger            <chr> "Alone", "Friend(s)", "Friend(s)", "Friend(s)", "~
## $ weather              <chr> "Sunny", "Sunny", "Sunny", "Sunny", "Sunny", "Sun~
## $ temperature          <int> 55, 80, 80, 80, 80, 80, 55, 80, 80, 80, 80, 55, 5~
## $ time                 <chr> "2PM", "10AM", "10AM", "2PM", "2PM", "6PM", "2PM"~
## $ coupon               <chr> "Restaurant(<20)", "Coffee House", "Carry out & T~
## $ expiration           <chr> "1d", "2h", "2h", "2h", "1d", "2h", "1d", "2h", "~
## $ gender               <chr> "Female", "Female", "Female", "Female", "Female",~
## $ age                  <chr> "21", "21", "21", "21", "21", "21", "21", "21", "~
## $ maritalStatus        <chr> "Unmarried partner", "Unmarried partner", "Unmarr~
## $ has_children         <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
## $ education            <chr> "Some college - no degree", "Some college - no de~
## $ occupation           <chr> "Unemployed", "Unemployed", "Unemployed", "Unempl~
## $ income               <chr> "$37500 - $49999", "$37500 - $49999", "$37500 - $~
## $ car                  <chr> "", "", "", "", "", "", "", "", "", "", "", "", "~
## $ Bar                  <chr> "never", "never", "never", "never", "never", "nev~
## $ CoffeeHouse          <chr> "never", "never", "never", "never", "never", "nev~
## $ CarryAway            <chr> "", "", "", "", "", "", "", "", "", "", "", "", "~
## $ RestaurantLessThan20 <chr> "4~8", "4~8", "4~8", "4~8", "4~8", "4~8", "4~8", ~
## $ Restaurant20To50     <chr> "1~3", "1~3", "1~3", "1~3", "1~3", "1~3", "1~3", ~
## $ toCoupon_GEQ5min     <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
## $ toCoupon_GEQ15min    <int> 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1~
## $ toCoupon_GEQ25min    <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1~
## $ direction_same       <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0~
## $ direction_opp        <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1~
## $ Y                    <int> 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1~
```


```r
accept_coupon %>% 
  summary()
```

```
##  destination         passanger           weather           temperature  
##  Length:12684       Length:12684       Length:12684       Min.   :30.0  
##  Class :character   Class :character   Class :character   1st Qu.:55.0  
##  Mode  :character   Mode  :character   Mode  :character   Median :80.0  
##                                                           Mean   :63.3  
##                                                           3rd Qu.:80.0  
##                                                           Max.   :80.0  
##      time              coupon           expiration           gender         
##  Length:12684       Length:12684       Length:12684       Length:12684      
##  Class :character   Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character   Mode  :character  
##                                                                             
##                                                                             
##                                                                             
##      age            maritalStatus       has_children     education        
##  Length:12684       Length:12684       Min.   :0.0000   Length:12684      
##  Class :character   Class :character   1st Qu.:0.0000   Class :character  
##  Mode  :character   Mode  :character   Median :0.0000   Mode  :character  
##                                        Mean   :0.4141                     
##                                        3rd Qu.:1.0000                     
##                                        Max.   :1.0000                     
##   occupation           income              car                Bar           
##  Length:12684       Length:12684       Length:12684       Length:12684      
##  Class :character   Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character   Mode  :character  
##                                                                             
##                                                                             
##                                                                             
##  CoffeeHouse         CarryAway         RestaurantLessThan20 Restaurant20To50  
##  Length:12684       Length:12684       Length:12684         Length:12684      
##  Class :character   Class :character   Class :character     Class :character  
##  Mode  :character   Mode  :character   Mode  :character     Mode  :character  
##                                                                               
##                                                                               
##                                                                               
##  toCoupon_GEQ5min toCoupon_GEQ15min toCoupon_GEQ25min direction_same  
##  Min.   :1        Min.   :0.0000    Min.   :0.0000    Min.   :0.0000  
##  1st Qu.:1        1st Qu.:0.0000    1st Qu.:0.0000    1st Qu.:0.0000  
##  Median :1        Median :1.0000    Median :0.0000    Median :0.0000  
##  Mean   :1        Mean   :0.5615    Mean   :0.1191    Mean   :0.2148  
##  3rd Qu.:1        3rd Qu.:1.0000    3rd Qu.:0.0000    3rd Qu.:0.0000  
##  Max.   :1        Max.   :1.0000    Max.   :1.0000    Max.   :1.0000  
##  direction_opp          Y         
##  Min.   :0.0000   Min.   :0.0000  
##  1st Qu.:1.0000   1st Qu.:0.0000  
##  Median :1.0000   Median :1.0000  
##  Mean   :0.7852   Mean   :0.5684  
##  3rd Qu.:1.0000   3rd Qu.:1.0000  
##  Max.   :1.0000   Max.   :1.0000
```


Further, we will employ exploration tools from othe packages. The `introduce()` and `plot_intro()` functions from the `DataExplorer` package provides an overview and visualization of the data. Finally, produce a data exploration report using `DataExplorer`



```r
accept_coupon %>% 
  DataExplorer::introduce()
```

```
##    rows columns discrete_columns continuous_columns all_missing_columns
## 1 12684      26               18                  8                   0
##   total_missing_values complete_rows total_observations memory_usage
## 1                    0         12684             329784      2245920
```


```r
accept_coupon %>% 
  DataExplorer::plot_intro()
```

![](coupon_recommendation_files/figure-html/unnamed-chunk-7-1.png)<!-- -->


```r
accept_coupon %>% 
  DataExplorer::create_report()
```






Investigating the `income` variable, we can see that income are divided into ranges such that it can be considered as categorical values. However, later it must be transformed into factors with the correct order of categories.


```r
accept_coupon %>% 
  select(income) %>% 
  unique()
```

```
##               income
## 1    $37500 - $49999
## 23   $62500 - $74999
## 45   $12500 - $24999
## 67   $75000 - $87499
## 111  $50000 - $62499
## 177  $25000 - $37499
## 194  $100000 or More
## 216  $87500 - $99999
## 238 Less than $12500
```



Check for multicollinearity using `cor()` and visualize it using `corrplot()` of the `corrplot` package


```r
accept_coupon %>% 
  select_if(is.numeric) %>%
  cor() %>% 
  corrplot::corrplot(method = "ellipse")
```

```
## Warning in cor(.): the standard deviation is zero
```

![](coupon_recommendation_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

The direction_same and direction_opp is exactly the same. This will cause an error in the modeling process. We can safely remove either one of them. 

### 3. Data Cleaning

Looking again at the data, there are entries which have the value `""`.

Counting the number of `""` in the dataset, there are about 13,370 entries with "" values. These are supposed to be NAs.  


```r
length(grep("", accept_coupon))
```

```
## [1] 26
```


```r
length(accept_coupon[accept_coupon == ""])
```

```
## [1] 13370
```

Converting the `""` into `NA` and checking again the number of `NA`s in the dataset, we find the following:


```r
accept_coupon[accept_coupon == ""] <- NA
```


```r
accept_coupon %>% 
  introduce()
```

```
##    rows columns discrete_columns continuous_columns all_missing_columns
## 1 12684      26               18                  8                   0
##   total_missing_values complete_rows total_observations memory_usage
## 1                13370           108             329784      2245584
```
Visualizing the missng data can give us a better understanding of the missingness.


```r
accept_coupon %>% 
  vis_miss()
```

![](coupon_recommendation_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

Using the `vis_dat` function allows us to visualize whether the missing values are coming from categorical (character) variables or numerical variables. Below, we can see that there


```r
accept_coupon %>% 
  vis_dat()
```

![](coupon_recommendation_files/figure-html/unnamed-chunk-16-1.png)<!-- -->



```r
accept_coupon %>% 
  naniar::miss_var_summary()
```

```
## # A tibble: 26 x 3
##    variable             n_miss pct_miss
##    <chr>                 <int>    <dbl>
##  1 car                   12576   99.1  
##  2 CoffeeHouse             217    1.71 
##  3 Restaurant20To50        189    1.49 
##  4 CarryAway               151    1.19 
##  5 RestaurantLessThan20    130    1.02 
##  6 Bar                     107    0.844
##  7 destination               0    0    
##  8 passanger                 0    0    
##  9 weather                   0    0    
## 10 temperature               0    0    
## # ... with 16 more rows
```

Almost all of the values of the `car` variable is missing! Checking the nature of the variables, the entries do not have a meaningful pattern that can be significant in influencing the model. We can safely exclude the whole variable in the modeling process.


```r
unique(accept_coupon$car)
```

```
## [1] NA                                        
## [2] "Scooter and motorcycle"                  
## [3] "crossover"                               
## [4] "Mazda5"                                  
## [5] "do not drive"                            
## [6] "Car that is too old to install Onstar :D"
```

In the process of cleaning the dataset, it will be renamed as `coupon`. This will allow us to revisit the original data if there will be additional inquiries that we would like to perform.






Now we will remove the `car` variable (to eliminate missing values) and `direction_opp` variables (to remove multicollinearity)


```r
# removing the `car` variable
coupon <- accept_coupon %>% 
  select(-c(car,direction_opp))
```



```r
setdiff(colnames(accept_coupon), colnames(coupon))
```

```
## [1] "car"           "direction_opp"
```



```r
miss_var_summary(coupon)
```

```
## # A tibble: 24 x 3
##    variable             n_miss pct_miss
##    <chr>                 <int>    <dbl>
##  1 CoffeeHouse             217    1.71 
##  2 Restaurant20To50        189    1.49 
##  3 CarryAway               151    1.19 
##  4 RestaurantLessThan20    130    1.02 
##  5 Bar                     107    0.844
##  6 destination               0    0    
##  7 passanger                 0    0    
##  8 weather                   0    0    
##  9 temperature               0    0    
## 10 time                      0    0    
## # ... with 14 more rows
```



```r
coupon %>% 
  mutate(
    across(where(is.character), as_factor)
  ) %>% 
  str()
```

```
## 'data.frame':	12684 obs. of  24 variables:
##  $ destination         : Factor w/ 3 levels "No Urgent Place",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ passanger           : Factor w/ 4 levels "Alone","Friend(s)",..: 1 2 2 2 2 2 2 3 3 3 ...
##  $ weather             : Factor w/ 3 levels "Sunny","Rainy",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ temperature         : int  55 80 80 80 80 80 55 80 80 80 ...
##  $ time                : Factor w/ 5 levels "2PM","10AM","6PM",..: 1 2 2 1 1 3 1 2 2 2 ...
##  $ coupon              : Factor w/ 5 levels "Restaurant(<20)",..: 1 2 3 2 2 1 3 1 3 4 ...
##  $ expiration          : Factor w/ 2 levels "1d","2h": 1 2 2 2 1 2 1 2 2 1 ...
##  $ gender              : Factor w/ 2 levels "Female","Male": 1 1 1 1 1 1 1 1 1 1 ...
##  $ age                 : Factor w/ 8 levels "21","46","26",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ maritalStatus       : Factor w/ 5 levels "Unmarried partner",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ has_children        : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ education           : Factor w/ 6 levels "Some college - no degree",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ occupation          : Factor w/ 25 levels "Unemployed","Architecture & Engineering",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ income              : Factor w/ 9 levels "$37500 - $49999",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ Bar                 : Factor w/ 5 levels "never","less1",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ CoffeeHouse         : Factor w/ 5 levels "never","less1",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ CarryAway           : Factor w/ 5 levels "4~8","1~3","gt8",..: NA NA NA NA NA NA NA NA NA NA ...
##  $ RestaurantLessThan20: Factor w/ 5 levels "4~8","1~3","less1",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ Restaurant20To50    : Factor w/ 5 levels "1~3","less1",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ toCoupon_GEQ5min    : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ toCoupon_GEQ15min   : int  0 0 1 1 1 1 1 1 1 1 ...
##  $ toCoupon_GEQ25min   : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ direction_same      : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ Y                   : int  1 0 1 0 0 1 1 1 1 0 ...
```

The percentage of the missing values in the variables `CoffeeHouse`,  `CarryAway`, `Bar`, `Restaurant20To50`, and `Restaurant20To50` are at the most 1.8 percent of a variable. We can also safely drop the rows with `NA`s without significant effect on the model   


```r
dim(coupon)
```

```
## [1] 12684    24
```



```r
coupon <- coupon %>% 
  drop_na()
```


```r
12079 / 12684
```

```
## [1] 0.9523021
```


```r
coupon %>% 
  vis_miss()
```

![](coupon_recommendation_files/figure-html/unnamed-chunk-26-1.png)<!-- -->


```r
which(coupon == "", arr.ind = TRUE)
```

```
##      row col
```


More than 95% of the data is retained.

A summary of the numeric values of the dataset shows the following statistics. The outcome value `Y` shows that the proportion of "Yes" (`1`) values is 0.57. This is a good proportion of outcome values where one outcome is not predominant over the other which will lead to a uneven distribution of outcomes in training and testing data. 


```r
coupon %>% 
  select_if(is.numeric) %>% 
  summary()
```

```
##   temperature     has_children    toCoupon_GEQ5min toCoupon_GEQ15min
##  Min.   :30.00   Min.   :0.0000   Min.   :1        Min.   :0.0000   
##  1st Qu.:55.00   1st Qu.:0.0000   1st Qu.:1        1st Qu.:0.0000   
##  Median :80.00   Median :0.0000   Median :1        Median :1.0000   
##  Mean   :63.33   Mean   :0.4085   Mean   :1        Mean   :0.5612   
##  3rd Qu.:80.00   3rd Qu.:1.0000   3rd Qu.:1        3rd Qu.:1.0000   
##  Max.   :80.00   Max.   :1.0000   Max.   :1        Max.   :1.0000   
##  toCoupon_GEQ25min direction_same         Y         
##  Min.   :0.0000    Min.   :0.0000   Min.   :0.0000  
##  1st Qu.:0.0000    1st Qu.:0.0000   1st Qu.:0.0000  
##  Median :0.0000    Median :0.0000   Median :1.0000  
##  Mean   :0.1194    Mean   :0.2152   Mean   :0.5693  
##  3rd Qu.:0.0000    3rd Qu.:0.0000   3rd Qu.:1.0000  
##  Max.   :1.0000    Max.   :1.0000   Max.   :1.0000
```


Now we will convert the income into a factor with spectific order


```r
# Extracting the categories

coupon %>% 
  select(income) %>% 
  unique() %>% 
  pull()
```

```
## [1] "$62500 - $74999"  "$12500 - $24999"  "$75000 - $87499"  "$50000 - $62499" 
## [5] "$37500 - $49999"  "$25000 - $37499"  "$100000 or More"  "$87500 - $99999" 
## [9] "Less than $12500"
```


```r
# Converting into an ordered factor

coupon <- coupon %>% 
  mutate(income = factor(income,
            levels = c(
              "Less than $12500",
              "$12500 - $24999",
              "$25000 - $37499",
              "$37500 - $49999",
              "$50000 - $62499",
              "$62500 - $74999",
              "$75000 - $87499",
              "$87500 - $99999",
              "$100000 or More"
            )))
```


```r
# Converting into an ordered factor

levels(coupon$income) <- c("A", "B", "C", "D", "E", "F", "G", "H", "I")

levels(coupon$income)
```

```
## [1] "A" "B" "C" "D" "E" "F" "G" "H" "I"
```





Converting `Y` into a factor variable


```r
coupon$Y <- factor(coupon$Y)
```


Check if there is a class imbalance for the outcome variable `Y`


```r
coupon %>% 
  count(Y)
```

```
##   Y    n
## 1 0 5202
## 2 1 6877
```

Though the class imbalance is not great, a downsampling will be performed to improve modeling performance



```r
coupon %>% 
  DataExplorer::plot_bar()
```

![](coupon_recommendation_files/figure-html/unnamed-chunk-34-1.png)<!-- -->![](coupon_recommendation_files/figure-html/unnamed-chunk-34-2.png)<!-- -->![](coupon_recommendation_files/figure-html/unnamed-chunk-34-3.png)<!-- -->


```r
?DataExplorer::create_report
```

```
## starting httpd help server ... done
```




```r
coupon %>% 
  DataExplorer::create_report(output_file = "cleaned_report.html")
```






## C. Resampling and Data Preprocessing 

### 1. Splitting into training and test sets

We will use the `tidymodels` packages to conduct resampling and data preprocessing. 

To ensure reproducibility, the seed of the random number generator will be set to 45 and the dataset will be split into training and test sets using `initial_split`, `training`, and `testing` functions from the `rsample` package. 


```r
# Create the split
# proportion of training is 70%; strata = Y is the outcome variable that will be predicted 

set.seed(45)

coupon_split <- rsample::initial_split(coupon, # cleaned dataset
                                       prop = 0.7, # proportion of training data
                                       strata = Y) # outcome variable
```


```r
coupon_training <- coupon_split %>% 
  rsample::training()

coupon_test <- coupon_split %>% 
  rsample::testing()
```


### 2.  Creating cross-validation folds

K-cross validation folds will be developed to help validate the best model.

The are 12684 instances for cleaned data set (after removing missing data). A 5-fold cross validation with approximately 2537 instances for each fold is reasonable.

Using the `vfold_cv` function of `rsample`


```r
set.seed(45)
coupon_folds <- rsample::vfold_cv(coupon_training, # training dataset
                                  v = 5, # number of folds
                                  strata = Y) # outcome variable
```

The result is a nested tibble (a special data frame that allows list entries). 


```r
coupon_folds
```

```
## #  5-fold cross-validation using stratification 
## # A tibble: 5 x 2
##   splits              id   
##   <list>              <chr>
## 1 <split [6762/1692]> Fold1
## 2 <split [6763/1691]> Fold2
## 3 <split [6763/1691]> Fold3
## 4 <split [6764/1690]> Fold4
## 5 <split [6764/1690]> Fold5
```

### 3. Feature engineering

Next, we will conduct feature engineering using `recipes` package to transform feature variables suitable for machine learning.

The following steps will be performed to build the data preprocessing recipe:
1. Scale numerical values
2. Convert character features into factors
3. Convert factors into dummy variables


```r
# library(themis)
```



```r
set.seed(45)
coupon_recipe <- recipe(Y ~ .,
                        data = coupon_training) %>% 
  # Downsampling
  themis::step_downsample(Y) %>% 
  # Scale numerical variables
  #step_normalize(all_numeric(), -all_outcomes()) %>% 
  # Convert character features into factors
  step_novel(all_nominal(), -all_outcomes()) %>% 
  # Convert factors into dummy variables 
  step_dummy(all_nominal(), -all_outcomes())
```

```
## Registered S3 methods overwritten by 'themis':
##   method                  from   
##   bake.step_downsample    recipes
##   bake.step_upsample      recipes
##   prep.step_downsample    recipes
##   prep.step_upsample      recipes
##   tidy.step_downsample    recipes
##   tidy.step_upsample      recipes
##   tunable.step_downsample recipes
##   tunable.step_upsample   recipes
```


```r
coupon_recipe
```

```
## Data Recipe
## 
## Inputs:
## 
##       role #variables
##    outcome          1
##  predictor         23
## 
## Operations:
## 
## Down-sampling based on Y
## Novel factor level assignment for all_nominal(), -all_outcomes()
## Dummy variables from all_nominal(), -all_outcomes()
```


## D. Specifying models and the workflow

In this classification problem we will use the following algorithms in R tha are accessible in the `tidymodels` interface:

* Logistic regression
* Decision trees
* Random forests
* Boosted trees



### 1. Specifying the models


Specify logistic regression - glm engine


```r
spec_log_reg_glm <- logistic_reg() %>% 
  set_engine(engine = "glm") %>% 
  set_mode("classification")

spec_log_reg_glm
```

```
## Logistic Regression Model Specification (classification)
## 
## Computational engine: glm
```


Specify decision trees model


```r
library(rpart)
```

```
## 
## Attaching package: 'rpart'
```

```
## The following object is masked from 'package:dials':
## 
##     prune
```

```r
spec_dt <- decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

spec_dt
```

```
## Decision Tree Model Specification (classification)
## 
## Computational engine: rpart
```


Specify random forest model


```r
library(ranger)

spec_rf <- rand_forest() %>% 
  set_engine(engine = "ranger", importance = "impurity") %>% 
  set_mode("classification")

spec_rf
```

```
## Random Forest Model Specification (classification)
## 
## Engine-Specific Arguments:
##   importance = impurity
## 
## Computational engine: ranger
```


Specify boosted tree model


```r
library(xgboost)
```

```
## 
## Attaching package: 'xgboost'
```

```
## The following object is masked from 'package:dplyr':
## 
##     slice
```

```r
spec_xgboost <- boost_tree() %>% 
  set_engine(engine = "xgboost") %>% 
  set_mode("classification")

spec_xgboost
```

```
## Boosted Tree Model Specification (classification)
## 
## Computational engine: xgboost
```







### 2. Initialize the workflows and choose model


To streamline the model process, the `workflow()` function from the `workflows` package will be used to combine the specified models and the data preprocessing recipe.


```r
library(workflows)
```


```r
?`workflows-package`
```


logistic regression (glm engine) workflow


```r
glm_workflow <- workflow() %>% 
  add_model(spec_log_reg_glm) %>% 
  add_recipe(recipe = coupon_recipe)

glm_workflow
```

```
## == Workflow ====================================================================
## Preprocessor: Recipe
## Model: logistic_reg()
## 
## -- Preprocessor ----------------------------------------------------------------
## 3 Recipe Steps
## 
## * step_downsample()
## * step_novel()
## * step_dummy()
## 
## -- Model -----------------------------------------------------------------------
## Logistic Regression Model Specification (classification)
## 
## Computational engine: glm
```


```r
glm_metrics <- glm_workflow %>% 
  fit_resamples(coupon_folds) %>% 
  collect_metrics()
```

```
## ! Fold1: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold2: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold3: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold4: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold5: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```r
glm_metrics
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
## 1 accuracy binary     0.668     5 0.00663 Preprocessor1_Model1
## 2 roc_auc  binary     0.729     5 0.00567 Preprocessor1_Model1
```



decision tree workflow


```r
dt_workflow <- workflow() %>% 
  add_model(spec_dt) %>% 
  add_recipe(recipe = coupon_recipe)

dt_workflow
```

```
## == Workflow ====================================================================
## Preprocessor: Recipe
## Model: decision_tree()
## 
## -- Preprocessor ----------------------------------------------------------------
## 3 Recipe Steps
## 
## * step_downsample()
## * step_novel()
## * step_dummy()
## 
## -- Model -----------------------------------------------------------------------
## Decision Tree Model Specification (classification)
## 
## Computational engine: rpart
```


```r
?tune::fit_resamples
```




```r
dt_metrics <- dt_workflow %>% 
  fit_resamples(coupon_folds) %>% 
  collect_metrics()

dt_metrics
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
## 1 accuracy binary     0.674     5 0.00333 Preprocessor1_Model1
## 2 roc_auc  binary     0.703     5 0.00475 Preprocessor1_Model1
```



```r
rf_workflow <- workflow() %>% 
  add_model(spec_rf) %>% 
  add_recipe(recipe = coupon_recipe)

rf_workflow
```

```
## == Workflow ====================================================================
## Preprocessor: Recipe
## Model: rand_forest()
## 
## -- Preprocessor ----------------------------------------------------------------
## 3 Recipe Steps
## 
## * step_downsample()
## * step_novel()
## * step_dummy()
## 
## -- Model -----------------------------------------------------------------------
## Random Forest Model Specification (classification)
## 
## Engine-Specific Arguments:
##   importance = impurity
## 
## Computational engine: ranger
```




```r
rf_metrics <- rf_workflow %>% 
  fit_resamples(coupon_folds) %>% 
  collect_metrics()

rf_metrics
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
## 1 accuracy binary     0.732     5 0.00719 Preprocessor1_Model1
## 2 roc_auc  binary     0.804     5 0.00867 Preprocessor1_Model1
```



```r
xg_workflow <- workflow() %>% 
  add_model(spec_xgboost) %>% 
  add_recipe(recipe = coupon_recipe)

xg_workflow
```

```
## == Workflow ====================================================================
## Preprocessor: Recipe
## Model: boost_tree()
## 
## -- Preprocessor ----------------------------------------------------------------
## 3 Recipe Steps
## 
## * step_downsample()
## * step_novel()
## * step_dummy()
## 
## -- Model -----------------------------------------------------------------------
## Boosted Tree Model Specification (classification)
## 
## Computational engine: xgboost
```


```r
xg_metrics <- xg_workflow %>% 
  fit_resamples(coupon_folds) %>% 
  collect_metrics()

xg_metrics
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
## 1 accuracy binary     0.716     5 0.00470 Preprocessor1_Model1
## 2 roc_auc  binary     0.782     5 0.00625 Preprocessor1_Model1
```





```r
all_metrics <- bind_rows("log_regression" = glm_metrics,
                     "decision_trees" = dt_metrics,
                     "random_forests" = rf_metrics,
                     "xg_boost" = xg_metrics,
                     .id = "models")

all_metrics
```

```
## # A tibble: 8 x 7
##   models         .metric  .estimator  mean     n std_err .config             
##   <chr>          <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
## 1 log_regression accuracy binary     0.668     5 0.00663 Preprocessor1_Model1
## 2 log_regression roc_auc  binary     0.729     5 0.00567 Preprocessor1_Model1
## 3 decision_trees accuracy binary     0.674     5 0.00333 Preprocessor1_Model1
## 4 decision_trees roc_auc  binary     0.703     5 0.00475 Preprocessor1_Model1
## 5 random_forests accuracy binary     0.732     5 0.00719 Preprocessor1_Model1
## 6 random_forests roc_auc  binary     0.804     5 0.00867 Preprocessor1_Model1
## 7 xg_boost       accuracy binary     0.716     5 0.00470 Preprocessor1_Model1
## 8 xg_boost       roc_auc  binary     0.782     5 0.00625 Preprocessor1_Model1
```


```r
all_metrics %>% 
  filter(.metric == "roc_auc") %>% 
  slice_max(mean)
```

```
## # A tibble: 1 x 7
##   models         .metric .estimator  mean     n std_err .config             
##   <chr>          <chr>   <chr>      <dbl> <int>   <dbl> <chr>               
## 1 random_forests roc_auc binary     0.804     5 0.00867 Preprocessor1_Model1
```


```r
all_metrics %>% 
  filter(.metric == "roc_auc") %>% 
  ggplot(aes(mean))
```



## E. Tuning the hyperparameters


### 1. Identify the hyperparameters of the random forest model

First we identify the hyperparameters that need to be tuned

The arguments for the `rand_forest()` interface are the following:


```r
?parsnip::rand_forest
```

* `mtry`: The number of predictors that will be randomly sampled at each split when creating the tree models.

* `trees`: The number of trees contained in the ensemble.

* `min_n`: The minimum number of data points in a node that are required for the node to be split further.


We will specify a random forest model with the parameters set to `tune()`. `tune()` is a placeholder function for the argument values that are to be tuned.


```r
spec_rf_tune <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

spec_rf_tune
```

```
## Random Forest Model Specification (classification)
## 
## Main Arguments:
##   mtry = tune()
##   trees = tune()
##   min_n = tune()
## 
## Computational engine: ranger
```


To know more about the parameters that will be tuned, we will pass the specified model to the `parameters()` function of the `dials` package


```r
?dials::parameters
```



```r
dials::parameters(spec_rf_tune)
```

```
## Collection of 3 parameters for tuning
## 
##  identifier  type    object
##        mtry  mtry nparam[?]
##       trees trees nparam[+]
##       min_n min_n nparam[+]
## 
## Model parameters needing finalization:
##    # Randomly Selected Predictors ('mtry')
## 
## See `?dials::finalize` or `?dials::update.parameters` for more information.
```

We can see that the `mtry` parameters needs finalization. To be able to generate values for `mtry` in the tuning grid, the upper bound must be determined, which is data dependent. 

We pass the predictors of the dataset and `mtry()` to the `finalize()` function from the `dials` package. It will return the upper bound for the 


```r
?dials::finalize
```


```r
coupon_pred <- coupon %>% 
  select(-Y)

dials::finalize(mtry(), coupon_pred)
```

```
## # Randomly Selected Predictors (quantitative)
## Range: [1, 23]
```

The value 23 will be used as the upper bound for the `mtry` parameter in creating a random grid,




### 2. Creating the tuning grid

After identifying and setting up the parameters for tuning, a grid of tuning parameters will be created usng the `grid_random()` function of the `dials` package.

The `grid_random()` will generate a number values for the tuning parameters randomly. Aside from reduced time in tuning the hyperparameters as compared to a systematic regular grid, randomly selecting the tuning parameters can improve chance of identifying the  values for the hyperparameters that will give the optimal results for the target metrics.



```r
# Create grids of tuning parameters

# Random and regular grids can be created for any number of parameter objects.

?dials::grid_random
```

`size`	
A single integer for the total number of parameter value combinations returned for the random grid. If duplicate combinations are generated from this size, the smaller, unique set is returned.

For the size we will use size = 5, as this is for demonstration purposes. Using the `grid_random()` function and passing a range of values for the `mtry` parameter, setting the upper bound value to 23, which was determined earlier using `finalize()`.


```r
rf_grid <- grid_random(mtry(range = c(1, 23)),
                        trees(),
                        min_n(),
                        size = 5)
```


```r
glimpse(rf_grid)
```

```
## Rows: 5
## Columns: 3
## $ mtry  <int> 21, 18, 15, 7, 6
## $ trees <int> 1290, 1691, 794, 1587, 369
## $ min_n <int> 2, 18, 15, 22, 8
```


### 3. Creating the tuning workflow and tuning the hyperparameters

To tune the hyperparameters, a workflow will also be created using the specified random forest model with parameters for tuning and the `coupon_recipe`.   


```r
rf_tune_wf <- workflow() %>% 
  add_recipe(coupon_recipe) %>% 
  add_model(spec_rf_tune)

rf_tune_wf
```

```
## == Workflow ====================================================================
## Preprocessor: Recipe
## Model: rand_forest()
## 
## -- Preprocessor ----------------------------------------------------------------
## 3 Recipe Steps
## 
## * step_downsample()
## * step_novel()
## * step_dummy()
## 
## -- Model -----------------------------------------------------------------------
## Random Forest Model Specification (classification)
## 
## Main Arguments:
##   mtry = tune()
##   trees = tune()
##   min_n = tune()
## 
## Computational engine: ranger
```

We can now tune the hyperparameters using `tune_grid()`. For reproducibility, we will also set the random seed to 45.

The tuning duration for the random forest model using the 5 sets (rows) of hyperparameter values will be measured using the `tictoc` package. Knowing the duration of the tuning can provide a rough estimate on the length of time it would take if we will increase the value of `size` argument of the `grid_random()` function to generate a tuning grid. 



```r
set.seed(45)

tictoc::tic()

rf_coupon_results <- tune_grid(
    rf_tune_wf,
    resamples = coupon_folds,
    grid = rf_grid
)

tictoc::toc()
```

```
## 414.76 sec elapsed
```


## F. Finalizing the model and Evaluating with the Unseen Testing Data 


### 1. Selecting the best model  

The results of the hyperparameter tuning is placed is stored in `rf_coupon_results`. Using the `collect_metrics()` we will extract the metrics to evaluate the best set of hyperparameter values that will provide the best value for the target metrics.


```r
rf_coupon_results %>% 
  collect_metrics()
```

```
## # A tibble: 10 x 9
##     mtry trees min_n .metric  .estimator  mean     n std_err .config            
##    <int> <int> <int> <chr>    <chr>      <dbl> <int>   <dbl> <chr>              
##  1    21  1290     2 accuracy binary     0.734     5 0.00661 Preprocessor1_Mode~
##  2    21  1290     2 roc_auc  binary     0.809     5 0.00815 Preprocessor1_Mode~
##  3    18  1691    18 accuracy binary     0.730     5 0.00439 Preprocessor1_Mode~
##  4    18  1691    18 roc_auc  binary     0.800     5 0.00812 Preprocessor1_Mode~
##  5    15   794    15 accuracy binary     0.731     5 0.00593 Preprocessor1_Mode~
##  6    15   794    15 roc_auc  binary     0.801     5 0.00807 Preprocessor1_Mode~
##  7     7  1587    22 accuracy binary     0.726     5 0.00702 Preprocessor1_Mode~
##  8     7  1587    22 roc_auc  binary     0.794     5 0.00847 Preprocessor1_Mode~
##  9     6   369     8 accuracy binary     0.729     5 0.00709 Preprocessor1_Mode~
## 10     6   369     8 roc_auc  binary     0.799     5 0.00829 Preprocessor1_Mode~
```



```r
rf_coupon_results %>% 
  collect_metrics(summarize = FALSE)
```

```
## # A tibble: 50 x 8
##    id     mtry trees min_n .metric  .estimator .estimate .config             
##    <chr> <int> <int> <int> <chr>    <chr>          <dbl> <chr>               
##  1 Fold1    21  1290     2 accuracy binary         0.729 Preprocessor1_Model1
##  2 Fold1    21  1290     2 roc_auc  binary         0.799 Preprocessor1_Model1
##  3 Fold2    21  1290     2 accuracy binary         0.723 Preprocessor1_Model1
##  4 Fold2    21  1290     2 roc_auc  binary         0.794 Preprocessor1_Model1
##  5 Fold3    21  1290     2 accuracy binary         0.759 Preprocessor1_Model1
##  6 Fold3    21  1290     2 roc_auc  binary         0.836 Preprocessor1_Model1
##  7 Fold4    21  1290     2 accuracy binary         0.725 Preprocessor1_Model1
##  8 Fold4    21  1290     2 roc_auc  binary         0.796 Preprocessor1_Model1
##  9 Fold5    21  1290     2 accuracy binary         0.732 Preprocessor1_Model1
## 10 Fold5    21  1290     2 roc_auc  binary         0.819 Preprocessor1_Model1
## # ... with 40 more rows
```


We will use the `roc_auc` metric as the criterion for selecting the set of hyperparameter values.

First, we will display the minimum, median, and maximum values of the `roc_auc` for each set of hyperparameter combinations to show the variation and spread of the values.


```r
rf_coupon_results %>% 
  collect_metrics(summarize = FALSE) %>% 
  filter(.metric == "roc_auc") %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))
```

```
## # A tibble: 5 x 4
##   id    min_roc_auc median_roc_auc max_roc_auc
##   <chr>       <dbl>          <dbl>       <dbl>
## 1 Fold1       0.781          0.789       0.799
## 2 Fold2       0.785          0.788       0.794
## 3 Fold3       0.820          0.827       0.836
## 4 Fold4       0.776          0.786       0.796
## 5 Fold5       0.808          0.811       0.819
```


To view which one is the best model, the `show_best()` function is used. It will return a tibble with the sorted values starting from best results.  



```r
# show_best {tune} - Investigate best tuning parameters

?tune::show_best
```



```r
rf_coupon_results %>% 
  show_best(metric = 'roc_auc', n = 5)
```

```
## # A tibble: 5 x 9
##    mtry trees min_n .metric .estimator  mean     n std_err .config             
##   <int> <int> <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>               
## 1    21  1290     2 roc_auc binary     0.809     5 0.00815 Preprocessor1_Model1
## 2    15   794    15 roc_auc binary     0.801     5 0.00807 Preprocessor1_Model3
## 3    18  1691    18 roc_auc binary     0.800     5 0.00812 Preprocessor1_Model2
## 4     6   369     8 roc_auc binary     0.799     5 0.00829 Preprocessor1_Model5
## 5     7  1587    22 roc_auc binary     0.794     5 0.00847 Preprocessor1_Model4
```


The best model will be selected using `select_best` function and stored to a variable `best_rf_coupon_model`.


```r
best_rf_coupon_model <- rf_coupon_results %>% 
  select_best(metric = 'roc_auc')

best_rf_coupon_model
```

```
## # A tibble: 1 x 4
##    mtry trees min_n .config             
##   <int> <int> <int> <chr>               
## 1    21  1290     2 Preprocessor1_Model1
```


### 2. Finalizing the model workflow

The `best_rf_coupon_model`, which is a set of hyperparameters in the tuning grid that gave the highest `roc_auc` value, will be used to update the final model workflow. 

Using the `finalize_workflow` function we will finalize the random forest workflow:


```r
final_coupon_wf <- rf_tune_wf %>% 
  finalize_workflow(best_rf_coupon_model)

final_coupon_wf
```

```
## == Workflow ====================================================================
## Preprocessor: Recipe
## Model: rand_forest()
## 
## -- Preprocessor ----------------------------------------------------------------
## 3 Recipe Steps
## 
## * step_downsample()
## * step_novel()
## * step_dummy()
## 
## -- Model -----------------------------------------------------------------------
## Random Forest Model Specification (classification)
## 
## Main Arguments:
##   mtry = 21
##   trees = 1290
##   min_n = 2
## 
## Computational engine: ranger
```


### 3. Evaluating the model using the testing data


Finally, the tuned random forest model will be fit to the unseen testing dataset. This will be done through the `last_fit()` function, which also passes the split dataset containing both training and testing data.


```r
coupon_final_fit <- final_coupon_wf %>% 
  last_fit(split = coupon_split)
```


```r
coupon_final_fit
```

```
## # Resampling results
## # Manual resampling 
## # A tibble: 1 x 6
##   splits        id          .metrics      .notes      .predictions     .workflow
##   <list>        <chr>       <list>        <list>      <list>           <list>   
## 1 <split [8454~ train/test~ <tibble [2 x~ <tibble [0~ <tibble [3,625 ~ <workflo~
```


```r
coupon_final_fit %>% 
  collect_metrics()
```

```
## # A tibble: 2 x 4
##   .metric  .estimator .estimate .config             
##   <chr>    <chr>          <dbl> <chr>               
## 1 accuracy binary         0.754 Preprocessor1_Model1
## 2 roc_auc  binary         0.829 Preprocessor1_Model1
```

Compared to the `roc_auc` value of 0.809 of the model using the training data, the `roc_auc` value of 0.829 of the testing data is even better!

















