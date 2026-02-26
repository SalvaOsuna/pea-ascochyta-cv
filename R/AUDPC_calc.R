# 1. Install necessary packages
# install.packages(c("tidyverse", "agricolae"))

library(tidyverse)
library(agricolae)

# 2. Load the Python-generated CSV
pheno_data <- read.csv("pea_phenotype_results.csv", sep = ";")

# 3. Clean and prepare the data
pheno_clean <- pheno_data %>%
  # Ensure DPI is recognized as a numeric time variable
  mutate(
    DPI = as.numeric(DPI),
    Genotype = as.factor(Genotype),
    Replicate = as.factor(Replicate),
    
    # Calculate Total Disease Severity (Necrosis + Chlorosis)
    Total_Disease = Necrosis_Perc + Chlorosis_Perc
  ) %>%
  # Crucial: Sort the data so time points are in chronological order for each plant
  arrange(Genotype, Replicate, DPI)

# 4. Calculate AUDPC across the time series (Robust Version)
audpc_results <- pheno_clean %>%
  # Group by each specific experimental unit
  group_by(Genotype, Replicate) %>%
  summarise(
    # Only calculate if there are at least 2 time points available
    AUDPC_Total     = if(n() > 1) audpc(Total_Disease, DPI) else NA,
    AUDPC_Necrosis  = if(n() > 1) audpc(Necrosis_Perc, DPI) else NA,
    AUDPC_Chlorosis = if(n() > 1) audpc(Chlorosis_Perc, DPI) else NA,
    
    # Extract the maximum severity reached
    Max_Total_Severity = max(Total_Disease, na.rm = TRUE),
    
    # Track how many days were actually measured for this plant
    Days_Measured = n(),
    
    .groups = "drop"
  )

# --- Check the missing data ---
# This will quickly show you which plants are missing time points
missing_data_check <- audpc_results %>%
  filter(Days_Measured < 4)

print("Experimental units missing time points:")
print(missing_data_check)

# 5. Calculate Genotype Means (Optional, for an overarching summary)
genotype_means <- audpc_results %>%
  group_by(Genotype) %>%
  summarise(
    Mean_AUDPC_Total    = mean(AUDPC_Total, na.rm = TRUE),
    Mean_AUDPC_Necrosis = mean(AUDPC_Necrosis, na.rm = TRUE),
    SE_AUDPC_Total      = sd(AUDPC_Total, na.rm = TRUE) / sqrt(n())
  )

# 6. View and Export
head(audpc_results)
write.csv(audpc_results, "AUDPC_Final_Calculations.csv", row.names = FALSE)
write.csv(genotype_means, "AUDPC_Genotype_Means.csv", row.names = FALSE)

#transformation####
library(tidyverse)
library(lme4)
library(car)      # For qqPlot

# 1. Create the transformed columns
audpc_results <- audpc_results %>%
  mutate(
    AUDPC_sqrt = sqrt(AUDPC_Total),
    AUDPC_log  = log1p(AUDPC_Total) # log1p is functionally log(x + 1)
  )

# 2. Fit the mixed models for raw, sqrt, and log data
# (Using a simpler formula just to quickly check residuals)
mod_raw  <- lmer(AUDPC_Total ~ Replicate + (1|Genotype), data = audpc_results)
mod_sqrt <- lmer(AUDPC_sqrt ~ Replicate + (1|Genotype), data = audpc_results)
mod_log  <- lmer(AUDPC_log ~ Replicate + (1|Genotype), data = audpc_results)

# 3. Plot the Residuals (QQ Plots)
par(mfrow = c(1, 3)) # Set up a 1x3 plotting grid

qqPlot(resid(mod_raw),  main = "Raw Data Residuals")
qqPlot(resid(mod_sqrt), main = "Square Root Residuals")
qqPlot(resid(mod_log),  main = "Log1p Residuals")

par(mfrow = c(1, 1)) # Reset grid

#The Replicate Consistency Boxplot####
library(tidyverse)
library(lme4)     # For calculating BLUPs
library(ggpubr)   # Optional: Makes ggplot themes look publication-ready

# 1. Boxplot: AUDPC across Replicates
ggplot(audpc_results, aes(x = Replicate, y = AUDPC_Total, fill = Replicate)) +
  geom_boxplot(alpha = 0.7, outlier.color = "red") +
  theme_minimal() +
  labs(
    title = "Ascochyta AUDPC Distribution Across Replicates",
    x = "Experimental Replicate",
    y = "Total AUDPC"
  ) +
  theme(legend.position = "none")

#Calculating the Genotype BLUPs####
# Ensure variables are factors
audpc_results$Genotype <- as.factor(audpc_results$Genotype)
audpc_results$Replicate <- as.factor(audpc_results$Replicate)

# Fit the mixed-effects model
# Replicate is fixed, Genotype is random (1|Genotype)
blup_model <- lmer(AUDPC_sqrt ~ Replicate + (1|Genotype), data = audpc_results)

# Extract the random effects (BLUPs)
raw_blups <- ranef(blup_model)$Genotype

# To make BLUPs biologically interpretable on the original AUDPC scale, 
# we add the model's grand mean (intercept) to each genotype's BLUP.
grand_mean <- fixef(blup_model)["(Intercept)"]

blup_df <- data.frame(
  Genotype = rownames(raw_blups),
  BLUP_Adjusted = raw_blups$`(Intercept)` + grand_mean
) %>%
  # Sort from most resistant (lowest AUDPC) to most susceptible (highest AUDPC)
  arrange(BLUP_Adjusted)

head(blup_df) # View the most resistant genotypes
write.csv(blup_df, "AUDPC_Total_BLUP.csv", row.names = FALSE)

#heritability####
library(lme4)

# 1. Extract the variance components from your mixed model
var_comps <- as.data.frame(VarCorr(blup_model))

# 2. Isolate Genetic Variance (Vg) and Residual Variance (Ve)
Vg <- var_comps$vcov[var_comps$grp == "Genotype"]
Ve <- var_comps$vcov[var_comps$grp == "Residual"]

# 3. Calculate the harmonic mean of replicates (r) to handle any missing data
# Count how many replicates survived the AUDPC calculation for each genotype
reps_per_geno <- audpc_results %>%
  group_by(Genotype) %>%
  summarise(count = n(), .groups = "drop")

# Harmonic mean formula: length(x) / sum(1/x)
r_harmonic <- length(reps_per_geno$count) / sum(1 / reps_per_geno$count)

# 4. Calculate H2
H2 <- Vg / (Vg + (Ve / r_harmonic)) #or just change this with the # of replicates
 H2 <- Vg / (Vg + (Ve / 3))

# Print the result
cat("\n--- Variance Components ---\n")
cat("Genetic Variance (Vg):", round(Vg, 2), "\n")
cat("Residual Variance (Ve):", round(Ve, 2), "\n")
cat("Effective Replicates (r):", round(r_harmonic, 2), "\n\n")

cat("--- Broad-Sense Heritability ---\n")
cat("H2 =", round(H2, 4), "\n")

#Plotting the Disease Progress Curves####
# 1. Select the extreme genotypes based on BLUPs
resistant_lines <- head(blup_df$Genotype, 3)
susceptible_lines <- tail(blup_df$Genotype, 3)

# 2. Filter the original longitudinal data and classify them
extreme_pheno <- pheno_clean %>%
  filter(Genotype %in% c(resistant_lines, susceptible_lines)) %>%
  mutate(
    Resistance_Class = ifelse(Genotype %in% resistant_lines, "Resistant", "Susceptible")
  ) %>%
  # Calculate the mean severity across the 3 replicates for each day
  group_by(Genotype, DPI, Resistance_Class) %>%
  summarise(Mean_Severity = mean(Total_Disease, na.rm = TRUE), .groups = "drop")

# 3. Create a subset of data just for the labels (placing them at the final time point)
label_data <- extreme_pheno %>%
  filter(DPI == max(DPI))

# 4. Plot the curves with Genotype labels
ggplot(extreme_pheno, aes(x = DPI, y = Mean_Severity, group = Genotype, color = Resistance_Class)) +
  geom_line(linewidth = 1.2, alpha = 0.8) +
  geom_point(size = 3) +
  
  # Add the text labels at the end of the lines
  geom_text(data = label_data, aes(label = Genotype), 
            hjust = -0.3, # Pushes the label slightly right of the final point
            fontface = "bold", 
            show.legend = FALSE) + # Prevents "a" from appearing in the legend
  
  scale_color_manual(values = c("Resistant" = "forestgreen", "Susceptible" = "firebrick")) +
  theme_minimal() +
  labs(
    title = "Ascochyta Disease Progress Curves",
    subtitle = "Top 3 Resistant vs. Most Susceptible Genotypes (Based on BLUPs)",
    x = "Days Post Inoculation (DPI)",
    y = "Mean Tissue Damage (%)"
  ) +
  # Expand the X-axis slightly (e.g., up to 9) so the labels have room to render
  scale_x_continuous(breaks = c(1, 3, 5, 8), limits = c(1, 9))

