function DataLoader(dataset) {

    var that = this;
    that.dataset = dataset;

    // Request nodes
    that.manifest_node = null;
    that.idx_node = null;
    that.feature
    that.label_node = null;
    that.sample_node = null;
    that.embed_data_node = null;
    that.grid_data_node = null;
    that.boundary_data_node = null;
    that.entropy_data_node = null;
    that.prediction_data_node = null;
    that.focus_data_node = null;

    // Data storage
    that.TrainData = null;
    that.ValidData = null;
    that.TestData = null;
    that.EntropyData = null;
    that.PredictionData = null;

    // Define topological structure of data retrieval
    that._init = function() {
        var params = "?dataset=" + dataset;
        that.manifest_node = new request_node(ManifestApi + params, manifest_handler, "json", "GET");
        that.idx_node = new request_node(IdxApi + params, idx_handler, "json", "GET");
        that.idx_node.depend_on(that.manifest_node);
        // that.feature_node = new request_node(FeatureApi + params, feature_handler, "json", "GET");
        // that.feature_node.depend_on(that.idx_node);
        that.label_node = new request_node(LabelApi + params, label_handler, "json", "GET");
        that.label_node.depend_on(that.idx_node);
        // that.sample_node = new request_node(SampleApi + params, sample_handler, "json", "GET");
        // that.sample_node.depend_on(that.label_node);
        that.embed_data_node = new request_node(DataApi + params  + "&embed-method=tsne", embed_data_handler, "json", "GET");
        that.embed_data_node.depend_on(that.label_node);
        that.entropy_data_node = new request_node(EntropyApi + params, entropy_handler, "json", "GET");
        that.entropy_data_node.depend_on(that.embed_data_node);
        that.prediction_data_node = new request_node(PredictionApi + params, prediction_handler, "json", "GET");
        that.prediction_data_node.depend_on(that.embed_data_node);
        // default datatype = test
        that.grid_data_node = new request_node(GridLayoutApi + params + "&datatype=test&embed-method=tsne&left-x=0&top-y=0&width=1&height=1&distribution=test&node-id=-1", grid_handler, "json", "GET");
        that.grid_data_node.depend_on(that.entropy_data_node);
        that.grid_data_node.depend_on(that.prediction_data_node);
        // that.grid_data_node.depend_on(that.confidence_data_node);
        // that.grid_data_node.depend_on(that.sample_node);
        // that.grid_data_node.set_off();

        that.focus_data_node = new request_node(FocusApi + params, data => focus_handler(LensView, data), "json", "GET");
        that.focus_data_node.depend_on(that.grid_data_node);
        that.focus_data_node.set_off();
        // that.entropy_data_node.set_off();
    };

    that.init = function() {
        that._init()
    }.call();

    that.get_data_container = function(datatype) {
        switch (datatype) {
            case "train":
                return that.TrainData;
                break;
            case "valid":
                return that.ValidData;
                break;
            case "test":
                return that.TestData;
                break;
            case "all":
                return null; // perform addition operation here.
                break;
            default:
                return null;
                break;
        }

    };

}