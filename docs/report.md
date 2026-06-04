# Potential Questions
- Which columns have missing values?
              Mising Count  Missing Percent
Electrical               1         0.068493
MasVnrArea               8         0.547945
BsmtCond                37         2.534247
BsmtFinType1            37         2.534247
BsmtQual                37         2.534247
BsmtExposure            38         2.602740
BsmtFinType2            38         2.602740
GarageType              81         5.547945
GarageQual              81         5.547945
GarageFinish            81         5.547945
GarageCond              81         5.547945
GarageYrBlt             81         5.547945
LotFrontage            259        17.739726
FireplaceQu            690        47.260274
MasVnrType             872        59.726027
Fence                 1179        80.753425
Alley                 1369        93.767123
MiscFeature           1406        96.301370
PoolQC                1453        99.520548

- Should any columns be removed?

| Missing % | Rules to follow                            |
| --------- | ------------------------------------------ |
| < 5%      | Usually safe to impute                     |
| 5–20%     | Investigate and impute carefully           |
| 20–50%    | Consider feature importance before keeping |
| > 50%     | Drop the column                            |


I will attempt to follow the rules I put together in the table above and check the result
