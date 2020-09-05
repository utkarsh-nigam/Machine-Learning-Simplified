self.finalModelInput = pd.concat((self.inputFile[self.ratioFeaturesAvailable + self.ordinalFeaturesAvailable],
                  pd.get_dummies(self.inputFile[self.nominalFeaturesAvailable])), 1)

self.finalModelInput[self.ratioFeaturesAvailable] = self.modelDetails["standardScalerEncoder"].transform(
    self.finalModelInput[self.ratioFeaturesAvailable])

self.finalModelInput[self.ordinalFeaturesAvailable] = self.modelDetails["ordinalEncoder"].transform(self.finalModelInput[self.ordinalFeaturesAvailable])

self.availableFeatures = self.finalModelInput.columns.tolist()
self.missingFeatures = list(set(self.modelDetails["modelFeatures"]) - set(self.availableFeatures))
self.additionalFeatures = list(set(self.availableFeatures) - set(self.modelDetails["modelFeatures"]))

self.finalModelInput.drop(columns=self.additionalFeatures, inplace=True)
for colName in self.missingFeatures:
    self.finalModelInput[colName] = 0

self.finalModelInput = self.finalModelInput[self.modelDetails["modelFeatures"]]