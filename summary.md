# Filtering numerical columns

The target column is categorical (0 = Not Fraud, 1 = Fraud)

For each numerical feature (X_num) colum do a two-sample t-test to compare means using H0: mean_group_0 = mean_group_1
If the H0 is rejected, then kepp the comlumn, otherwise drop it.

Among the remaining ones, do the VIF analysis (don't include the target column)

# Filtering categorical columns

For each categorical feature column do a Chi2 test between the feature column and the target column. H0: both column are not related
If you reject the H0, keep the column.