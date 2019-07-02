# Some basic tests

DataManager = tools.dynamicImportDev('dataManager').Manager

# traininigDataFun, trainingData = tools.importData('training')
# trainingDataProvider = DataProvider(
#     data = traininigDataFun,
#     bz = env.training.bz,
#     stochasticSampling = False,
#     indexingShape = [trainingData.shape[0]]
# )

# print(np.array(trainingDataProvider(1)[0]).shape)

validationDataFun, validationData = tools.importData('validation')
print(len(validationData))

validationDataProvider = DataManager(
    data = validationDataFun,
    bz = env.training.bz,
    stochasticSampling = False,
    reshuffle = True,
    indexingShape = [len(validationData)]
)

print(np.array(validationDataProvider(1)[1]).shape)

# prove that reshouffling works by
# 1) showing that we get the same number of indexs as original indexs (minux batch size leftovers)
# 2) the data we get matches the index provided equivalents in the origina ordered data source 

indxs = []
data
for i, data, indx in validationDataProvider:
    print(np.array(data[1])[:, :, :, 0].shape, validationData[indx][:, :, :, 3].shape)
    print(np.mean(np.array(data[1])[:, :, :, 0] == validationData[indx][:, :, :, 3]))
    indxs += list(indx) 

len(len(validationData) - len(validationData)%env.training.bz)