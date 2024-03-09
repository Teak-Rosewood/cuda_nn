
export interface Layer {
    layerId: number
    units: number
    activation: string
}

export class ModelManager {
    private model_file: File | null;
    private model: Layer[];

    constructor() {
        this.model = []
        this.model_file = null;
    }

    updateModel(model: Layer[]) {
        this.model = model
    }

    getModel() {
        return this.model;
    }

    compileModel() {

    }

    async fitModel() {
        return this.model_file;
     }

    async getDatasetFeatures() {
        return ["feature1", "feature2", "feature3"]
    }

    async setPredictionFeature(feature: string) {
        return true;
    }   

    async uploadDataset(file: File) {
        return this.getDatasetFeatures();
    }
}