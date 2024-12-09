library(naniar)
library(dplyr)
library(ggplot2)
library(readxl)
library(mice)

# # from raw data to cleaned dataset, how many variables is deleted and its reason, 
# # how many variable is created and reason.

# replace NULL with NA-----------
data_clean <- read.csv("data/UTK_med_data_cleaned.csv",  header = TRUE)
data_clean_nan <- read.csv("data/UTK_med_data_cleaned_with_nan.csv",  header = TRUE)

data_clean[data_clean == ""] <- NA
data_clean_nan[data_clean_nan == ""] <- NA


rawdata <- read_excel("data/UTK_UTMC_OBGYN_DEIDENTIFIED_2023-4-26_Data.xlsx")
rawdata[rawdata == ""] <- NA
summary(rawdata)
# 
# # # visualize the missing percentage-----------
# # vis_miss(data_clean,warn_large_data = FALSE)
# vis_miss(data_clean_nan,warn_large_data = FALSE)
# # 
# # # downsampling
# # data_clean %>%
# #   slice(1:1000) %>%
# #   #vis_miss(warn_large_data = FALSE)+
# #   ggtitle("Missing Data Visualization for Cleaned Dataset") +  
# #   theme(axis.text = element_text(size = 4),  # Change the size of the axis text
# #         axis.title = element_text(size = 14), # Change the size of the axis titles
# #         plot.title = element_text(size = 16))
# # 
# data_clean_nan %>%
#   #slice(1:1000) %>%
#   vis_miss(warn_large_data = FALSE)+ggtitle("Missing Data Visualization for Cleaned Dataset with NAN") +
#   theme(axis.text = element_text(size = 4),  # Change the size of the axis text
#         axis.title = element_text(size = 14), # Change the size of the axis titles
#         plot.title = element_text(size = 16))
# 
# # 
# # 
# rawdata %>%
#   #slice(1:1000) %>%
#   vis_miss(warn_large_data = FALSE)+ggtitle("Missing Data Visualization for Raw Data") +
#   theme(axis.text = element_text(size = 4),  # Change the size of the axis text
#         axis.title = element_text(size = 14), # Change the size of the axis titles
#         plot.title = element_text(size = 16))


# manipulate reason for C section--------------------
# Define the mapping of reasons to categories
reason_map <- c(
  'rptelect' = 'Elective_Repeat',
  'elective' = 'Elective_Repeat',
  'primaryelective' = 'Elective_Repeat',
  'breech' = 'Fetal_Issues',
  'fetbrady' = 'Fetal_Issues',
  'fetalintoleranceoflabor' = 'Fetal_Issues',
  'funiccordpresentation' = 'Fetal_Issues',
  'macrosomia' = 'Fetal_Issues',
  'non-reassurringfetalheartratestatus' = 'Fetal_Issues',
  'nonreassuringfetalstatus' = 'Fetal_Issues',
  'fetalgrowthrestriction' = 'Fetal_Issues',
  'pih' = 'Maternal_Health',
  'proactiv' = 'Maternal_Health',
  'eclampsia' = 'Maternal_Health',
  'maternalaorticstenosis' = 'Maternal_Health',
  'acherpes' = 'Maternal_Health',
  'condylom' = 'Maternal_Health',
  'arrdscnt' = 'Labor_Issues',
  '2ndardil' = 'Labor_Issues',
  'faildscn' = 'Labor_Issues',
  'failuretoprogress' = 'Labor_Issues',
  'arrestofdilation' = 'Labor_Issues',
  'ftp' = 'Labor_Issues',
  'prodecel' = 'Labor_Issues',
  'protdesc' = 'Labor_Issues',
  'failuretodilate' = 'Labor_Issues',
  'arrestofdilatation' = 'Labor_Issues',
  'placprev' = 'Placenta_Cord_Issues',
  'abrplac' = 'Placenta_Cord_Issues',
  'cord_pro' = 'Placenta_Cord_Issues',
  'prolapsedcord' = 'Placenta_Cord_Issues',
  'suspectedplacentalabruption' = 'Placenta_Cord_Issues',
  'utrnscar' = 'Uterine_Issues',
  '?utrnrup' = 'Uterine_Issues',
  'possibleuterinerupture' = 'Uterine_Issues',
  'uterineseptum' = 'Uterine_Issues',
  'multgest' = 'Multiple_Gestations',
  'diditwins/breechpresentation' = 'Multiple_Gestations',
  'pretermlabor' = 'Preterm_Labor_Complications',
  'ppromat24.4wks,ptl,malpresentation' = 'Preterm_Labor_Complications',
  'prolongedruptureofmembranes' = 'Preterm_Labor_Complications',
  'srom' = 'Preterm_Labor_Complications',
  'nocsection' = 'Vag',
  'nonrefhr' = 'Other_Not_Specified',
  'other' = 'Other_Not_Specified',
  'indcfail' = 'Other_Not_Specified',
  'profetde' = 'Other_Not_Specified',
  'prolaten' = 'Other_Not_Specified',
  'xvrscmpl' = 'Other_Not_Specified',
  '>2csects' = 'Other_Not_Specified',
  'prvmyome' = 'Other_Not_Specified',
  'hellp' = 'Other_Not_Specified',
  'suspectedmacrosomia' = 'Other_Not_Specified',
  'pre-ewithoutseverefeatures' = 'Other_Not_Specified',
  'vaginalbleeding' = 'Other_Not_Specified',
  'hellp,severeiugrlessthan3%' = 'Other_Not_Specified'
)

# Apply the mapping to create a new column in the dataframe
data <- data_clean_nan %>%
  mutate(Reason.For.Csection.new = reason_map[Reason.For.Csection])
data <-  data%>% select(-Reason.For.Csection)



# combine low county into one category:others-----------
data_lowmiss <- data %>%
  group_by(County) %>%
  mutate(County = ifelse(n() == 1, 'Other', County)) %>%
  ungroup()

# combine low count provider into one category:others-----------
data_lowmiss <- data_lowmiss %>%
  group_by(Provider.On.Admission) %>%
  mutate(Provider.On.Admission = ifelse(n() == 1, 'Other', Provider.On.Admission)) %>%
  ungroup()



# delete Reason.For.Csection.new: too many missing values--------
data_lowmiss <- data_lowmiss %>% select(-Reason.For.Csection.new)


# delete the high missing percentage variables-----
######### Analysis based on cleand_NAN data
data_lowmiss <- data_lowmiss[,-1]
missing_percent <- miss_var_summary(data_lowmiss) 
high_missing_vars <- missing_percent %>%
  filter(pct_miss > 75) %>% # later a variable with 70% is still deleted
  pull(variable)# 10 variables has high missing values
# Remove columns with high percentage of missing values from the dataframe
data_lowmiss <- data_lowmiss %>% select(-one_of(high_missing_vars))

# delete some date columns-----------
# time_columns <- data_lowmiss %>%
#   select_if(~inherits(., c("Date", "POSIXct", "POSIXlt"))) %>%
#   names()
data_lowmiss <- data_lowmiss %>% select(-Membranes.Rupture.Time
                                        ,-Placenta.Delivery.Time, -Complete.Dilatation)# these 2 have already been used to calculate labor time in stage 2 and 3


# choose response variable---------

# 1. method of delivery (vag/csection)
# 2. emergency Csection or not
# 3. labor time in stage 2

#

# delete some unrelated variables----------
summary(data_lowmiss)
str(data_lowmiss)
# Function to display tables for character columns and statistics for numeric columns
summarize_dataframe <- function(df) {
  summary_list <- list()
  
  for (col_name in names(df)) {
    if (is.character(df[[col_name]])) {
      # Show table for character columns
      summary_list[[col_name]] <- table(df[[col_name]])
    } else if (is.numeric(df[[col_name]])) {
      # Show statistics for numeric columns
      summary_list[[col_name]] <- summary(df[[col_name]])
    }
  }
  
  return(summary_list)
}



features <- read.csv("data/variables_select_FINAL.csv",  header = TRUE,row.names = 1)
rows_with_N <- rownames(features)[apply(features[, 1:3], 1, function(row) all(row == "N"))]
data_lowmiss <- data_lowmiss %>% select(-one_of(rows_with_N))



summary_results <- summarize_dataframe(data_lowmiss)

# Print the summary results
for (name in names(summary_results)) {
  cat("\nColumn:", name, "\n")
  print(summary_results[[name]])
}



# set data type---------
num_columns <- names(data_lowmiss)[sapply(data_lowmiss, is.numeric)]
character_columns <- names(data_lowmiss)[sapply(data_lowmiss, is.character)]
logical_columns <- character_columns[31:76]
factor_columns <- setdiff(character_columns, logical_columns)
# factor_columns <- c(factor_columns,num_columns[c(2:5)])
# num_columns <- num_columns[-c(2:5)]
# logical_columns
# factor_columns
# num_columns
str(data_lowmiss)
write(logical_columns, file = "data/logical_columns.txt")
write(factor_columns, file = "data/factor_columns.txt")
write(num_columns, file = "data/num_columns.txt")


data_NA <- data_lowmiss
data_NA[factor_columns] <- lapply(data_NA[factor_columns], as.factor)
data_NA[logical_columns] <- lapply(data_NA[logical_columns], as.logical)
str(data_NA)


# data imputation -------------
data_NA %>%
  vis_miss(warn_large_data = FALSE)+ggtitle("Missing Data Visualization for Final Data") +  
  theme(axis.text = element_text(size = 4),  # Change the size of the axis text
        axis.title = element_text(size = 14), # Change the size of the axis titles
        plot.title = element_text(size = 16))

missing_percent <- miss_var_summary(data_NA) 


##### Complete case 
data_complete <- na.omit(data_NA)

# ##### mean-value imputation
# # mean
# replace_na_numeric <- function(df) {
#   df %>%
#     mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))
# }
# data_mean <- replace_na_numeric(data_lowmiss)
#
# # the most used one
# replace_na_categorical <- function(df) {
#   df %>%
#     mutate(across(where(is.character), ~ifelse(is.na(.), Mode(.), .)))
# }
#
# Mode <- function(x) {
#   ux <- unique(x)
#   ux[which.max(tabulate(match(x, ux)))]
# }
#
#
# data_mean <- replace_na_categorical(data_mean)
# #### Machine learning way




# #### MICE
# Initialize default methods
# delete county and provider for now since it has too many levels
# data_NA_mice <- data_NA %>% select(-County,-Provider.On.Admission)
# str(data_NA_mice)
# # Identify near-zero variance predictors
# library(caret)
# nzv <- nearZeroVar(data_NA_mice, saveMetrics = TRUE)
# print(nzv)


#
# imputation_methods <- make.method(data_NA_mice)
#
#
# # Specify methods for categorical variables
# imputation_methods["Age"] <- "pmm"  
# #imputation_methods["County"] <- "polyreg"   # Polytomous regression for categorical variables
# imputation_methods["Gravida"] <- "pmm" 
# imputation_methods["Para"] <- "pmm" 
# imputation_methods["Episiotomy"] <- "polyreg" 
# imputation_methods["Laceration.Extension"] <- "polyreg" 
# imputation_methods["Amniotic.Fluid.Color"] <- "polyreg" 
# imputation_methods["Attempt.to.Vbac"] <- "polyreg" 
# imputation_methods["Csection.Incidence"] <- "polyreg" 
# imputation_methods["Csection.Urgency"] <- "polyreg" 
# imputation_methods["Membranes.Rupture.Method"] <- "polyreg" 
# imputation_methods["Number.of.Babies.In.Womb"] <- "pmm" 
# imputation_methods["Oxytocin"] <- "polyreg" 
# imputation_methods["Delivery.Month"] <- "polyreg" 
# imputation_methods["Delivery.Year"] <- "pmm" 
# imputation_methods["Gestational.Status.A"] <- "polyreg" 
# imputation_methods["Born.In.Route.A"] <- "logreg" 
# imputation_methods["Infant.Cord.Vessels.A"] <- "pmm" 
# imputation_methods["Infant.Length.A"] <- "pmm" 
# imputation_methods["Fetal.Presentation.A"] <- "polyreg" 
# imputation_methods["Method.of.Delivery.A"] <- "logreg" 
# imputation_methods["Forceps.a"] <- "polyreg" 
# imputation_methods["Vacuum.Extraction.A"] <- "polyreg" 
# imputation_methods["VBAC.A"] <- "polyreg" 
# imputation_methods["Infant.Sex.A"] <- "logreg" 
# imputation_methods["Birthweight.A"] <- "pmm"  
# imputation_methods["GA.At.Birth.A"] <- "pmm" 
# imputation_methods["Cervical.Ripening.Agents"] <- "polyreg" 
# imputation_methods["Weight.Prepregnancy"] <- "pmm" 
# imputation_methods["Weight.Gain.Pounds"] <- "pmm" 
# imputation_methods["Preterm.Labor"] <- "logreg" 
# imputation_methods["Premature.Rom"] <- "logreg" 
# imputation_methods["Hypertensive.Disorders.Preg"] <- "logreg" 
# imputation_methods["Diabetes"] <- "logreg" 
# imputation_methods["Adm.Alcohol"] <- "logreg" 
# imputation_methods["Adm.Cigarettes"] <- "polyreg" 
# imputation_methods["Adm.Marijuana"] <- "logreg" 
# imputation_methods["Adm.Cocaine.Crack"] <- "logreg" 
# imputation_methods["Adm.Illicit.Drugs"] <- "logreg" 
# imputation_methods["OFC.At.Birth.Baby.A.In"] <- "pmm" 
# #imputation_methods["Provider.On.Admission"] <- "polyreg" 
# imputation_methods["Quantitative.Blood.Loss.DS"] <- "pmm" 
# imputation_methods["Ethnicity"] <- "polyreg" 
# imputation_methods["Weight.Delivery"] <- "pmm" 
# imputation_methods["BMI_delivery"] <- "pmm" 
# imputation_methods["BMI_prepregnancy"] <- "pmm" 
# imputation_methods["Labortime"] <- "pmm" 
# imputation_methods["delivery_anesthesia_epidural"] <- "logreg" 
# imputation_methods["delivery_anesthesia_spinal"] <- "logreg" 
# imputation_methods["delivery_anesthesia_local"] <- "logreg" 
# imputation_methods["delivery_anesthesia_pudenal"] <- "logreg" 
# imputation_methods["delivery_anesthesia_cse"] <- "logreg" 
# imputation_methods["delivery_anesthesia_general"] <- "logreg" 
# imputation_methods["delivery_anesthesia_intrathe"] <- "logreg" 
# imputation_methods["labor_anesthesia_epidural"] <- "logreg" 
# imputation_methods["labor_anesthesia_spinal"] <- "logreg" 
# imputation_methods["labor_anesthesia_cse"] <- "logreg" 
# imputation_methods["labor_anesthesia_general"] <- "logreg" 
# imputation_methods["labor_anesthesia_iv_sedation"] <- "logreg" 
# imputation_methods["laceration_type_none"] <- "logreg" 
# imputation_methods["laceration_type_vaginal"] <- "logreg" 
# imputation_methods["laceration_type_perineal"] <- "logreg" 
# imputation_methods["laceration_type_periuret"] <- "logreg" 
# imputation_methods["laceration_type_cervical"] <- "logreg" 
# imputation_methods["laceration_type_other"] <- "logreg" 
# imputation_methods["infant_complications_A_brady"] <- "logreg" 
# imputation_methods["infant_complications_A_acidosis"] <- "logreg" 
# imputation_methods["infant_complications_A_meconium"] <- "logreg" 
# imputation_methods["infant_complications_A_cord_pro"] <- "logreg" 
# imputation_methods["infant_complications_A_MLD"] <- "logreg" 
# imputation_methods["infant_complications_A_MVD"] <- "logreg" 
# imputation_methods["infant_complications_A_oligohyd"] <- "logreg" 
# imputation_methods["infant_complications_A_shol_dys"] <- "logreg" 
# imputation_methods["infant_complications_A_polyhydr"] <- "logreg" 
# imputation_methods["infant_complications_A_decvarib"] <- "logreg" 
# imputation_methods["infant_complications_A_tachy"] <- "logreg" 
# imputation_methods["infant_complications_A_other"] <- "logreg" 
# imputation_methods["reason_for_admission_Spon_Ab"] <- "logreg" 
# imputation_methods["reason_for_admission_Labor"] <- "logreg" 
# imputation_methods["reason_for_admission_PTL"] <- "logreg" 
# imputation_methods["reason_for_admission_Induce"] <- "logreg" 
# imputation_methods["reason_for_admission_Rept_CS"] <- "logreg" 
# imputation_methods["reason_for_admission_Prim_CS"] <- "logreg" 
# imputation_methods["reason_for_admission_ROM"] <- "logreg" 
# imputation_methods["reason_for_admission_Observe"] <- "logreg" 
# imputation_methods["reason_for_admission_other"] <- "logreg" 
# imputation_methods["race_black"] <- "logreg" 
# imputation_methods["race_white"] <- "logreg" 
# imputation_methods["race_pacific_islander"] <- "logreg" 
# imputation_methods["race_asian"] <- "logreg"
# imputation_methods["race_arabic"] <- "logreg" 
# imputation_methods["race_american_indian"] <- "logreg" 
# imputation_methods["race_other"] <- "logreg"



# Check the pattern of missing data
# print("Missing Data Pattern:")
# md.pattern(data_lowmiss)
str(data_NA)
# Perform multiple imputation <will only impute numerical data since no character imputation methods given>
dataset_mice <- mice(data_lowmiss, m = 5,maxit = 50,seed = 123)

#Print a summary of the imputation
summary(dataset_mice)

# Get the first imputed dataset
data_mice <- complete(dataset_mice, 1)

data_mice_num <- data_mice[,num_columns]
data_mice_chr <- data_mice[,character_columns]

# Function to calculate the mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Apply the mode imputation to the dataframe
replace_na_mode <- function(df) {
  df[] <- lapply(df, function(x) {
    if (any(is.na(x))) {
      x[is.na(x)] <- Mode(x)
    }
    return(x)
  })
  return(df)
}

# Example usage with data_cat dataframe
data_mice_chr <- replace_na_mode(data_mice_chr)

data_mice <- cbind(data_mice_num,data_mice_chr)

str(data_mice)



missing_mice <- miss_var_summary(data_mice)

data_mice %>%
  vis_miss(warn_large_data = FALSE)+ggtitle("Missing Data Visualization for Final Data") +  
  theme(axis.text = element_text(size = 4),  # Change the size of the axis text
        axis.title = element_text(size = 14), # Change the size of the axis titles
        plot.title = element_text(size = 16))



write.csv(data_NA, file = "./data/data_NA.csv", row.names = FALSE)
write.csv(data_complete, file = "./data/data_complete.csv", row.names = FALSE)
#write.csv(data_mean, file = "./data/data_mean.csv", row.names = FALSE)
write.csv(data_mice, file = "./data/data_mice.csv", row.names = FALSE)
data_complete <- read.csv("EHR_UTK/data/data_complete.csv",  header = TRUE)
data_mean <- read.csv("EHR_UTK/data/data_mean.csv",  header = TRUE)
data_ml <- read.csv("EHR_UTK/data/data_ml.csv",  header = TRUE)
data_mice <- read.csv("EHR_UTK/data/data_mice.csv",  header = TRUE)

### plot for response: emergency
Emer_com <- as.data.frame(table(data_complete$Csection.Urgency))
Emer_mean <- as.data.frame(table(data_mean$Csection.Urgency))
Emer_ml <- as.data.frame(table(data_ml$Csection.Urgency))
Emer_mice <- as.data.frame(table(data_mice$Csection.Urgency))

df_emer <- rbind(Emer_com,Emer_mean,Emer_ml,Emer_mice)
colnames(df_emer) <- c("Category","Count")
df_emer <- df_emer[df_emer$Category != "Schedule", ]
df_emer$Dataset <- rep(c("complete", "mean", "ML", "MICE"), each=3)
df_emer$count2 <- df_emer$Count
df_emer[4:9,4] <- ""



ggplot(df_emer, aes(x=factor(Category),y = Count, fill=Dataset))+
  geom_bar(stat="identity", position=position_dodge())+
  geom_text(aes(x = factor(Category),label=count2), 
            position = position_dodge(width = 1),
            vjust = -0.5, size = 3)+
  xlab("Method of Delivery") + ylab("Count")+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()


ggplot(df_emer, aes(x=factor(Category),y = Count, fill=Dataset))+
  geom_col(position=position_dodge())+
  xlab("Method of Delivery") + ylab("Count")+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()
