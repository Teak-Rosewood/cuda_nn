
export interface Layer {
    layerId: number
    units: number
    activation: string
}

export class ModelManager {
    private model_file: File | null;
    private model: Layer[];
    private prediction_var: string;

    constructor() {
        this.model = []
        this.model_file = null;
        this.prediction_var = "";
    }

    updateModel(model: Layer[]) {
        this.model = model
        console.log(this.model)
    }

    getModel() {
        return this.model;
    }

    async compileModel(layers: Layer[]) {
        const response = await fetch('http://localhost:5000/api/compile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ layers }),
        });
        const res = await response.json();
        return res.message;
    }

    async fitModel(epochs: number, layers: Layer[]) {
        const pred = this.prediction_var
        console.log(pred)
        const response = await fetch('http://localhost:5000/api/fit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ epochs, pred, layers }),
        });
        const res = await response.json();
        console.log("got res")
        return res;
    }

    async getDatasetFeatures() {
        return ["feature1", "feature2", "feature3"]
    }

    async setPredictionFeature(feature: string) {
        console.log(feature)
        this.prediction_var = feature;
        return true;
    }

    async uploadDataset(file: File) {
        const formData = new FormData();
        formData.append('file', file, 'data.csv');

        const response = await fetch('http://localhost:5000/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            return false;
        }
        const res = await response.json();
        return res.features;
    }
}