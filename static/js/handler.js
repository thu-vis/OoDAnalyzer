/**
 * Created by Changjian on 2018/12/17.
 */

var manifest_handler = function(data){
    ManifestData = data;
    TrainInstanceNum = data.TrainInstanceNum;
    ValidInstanceNum = data.ValidInstanceNum;
    TestInstanceNum = data.TestInstanceNum;
    FeatureNum = data.FeatureNum;
    LabelNames = data.LabelNames;
    LabelShown = Array(LabelNames.length).fill(true);
    TrainAcc = data["train-acc"];
    DataType = "test";
    document.getElementById("train-num").textContent = TrainInstanceNum;
    document.getElementById("test-num").textContent = TestInstanceNum;
    // document.getElementById("train-acc").textContent = TrainAcc.toFixed(2);
    legend_create();
};

var idx_handler = function(data){
    IdxData = data;
    let img_prefix = ImageApi + "?dataset=" + DatasetName;
    let thumbnail_prefix = ThumbnailApi + "?dataset=" + DatasetName;
    var raw_data_list = [data.train_idx, data.valid_idx, data.test_idx];
    Loader.TrainData = new DataContainer("train", img_prefix, thumbnail_prefix);
    Loader.ValidData = new DataContainer("valid", img_prefix, thumbnail_prefix);
    Loader.TestData = new DataContainer("test", img_prefix, thumbnail_prefix);
    var data_variables_list = [Loader.TrainData, Loader.ValidData, Loader.TestData];
    var datatype_list = ["train", "valid", "test"];
    for( var i = 0; i < raw_data_list.length; i++ ){
        var variable = data_variables_list[i];
        var idx_data = raw_data_list[i];
        var type = datatype_list[i];
        for ( var j = 0; j < idx_data.length; j++ ){
            var c = new DataCell(idx_data[j],
                variable.get_img_url_prefix(),
                variable.get_thumbnail_url_prefix(),
                variable.get_data_type());
            variable.append_datacell(c);
            c.set_datatype(type);
        }
    }
};

var embed_data_handler = function(data){
    EmbedData = data;
    var raw_data_list = [data.embed_X_train, data.embed_X_valid, data.embed_X_test];
    var data_variables_list = [Loader.TrainData, Loader.ValidData, Loader.TestData];
    for( var i = 0; i < raw_data_list.length; i++ ){
        var variable = data_variables_list[i];
        var embed_data = raw_data_list[i];
        for ( var j = 0; j < embed_data.length; j++ ){
            variable.get_cell(j).set_embed_x(embed_data[j]);
        }
    }
};

var feature_handler = function(data) {
    var raw_data_list = [data.X_train, data.X_valid, data.X_test];
    var data_variables_list = [Loader.TrainData, Loader.ValidData, Loader.TestData];
    for( var i = 0; i < raw_data_list.length; i++ ){
        var variable = data_variables_list[i];
        var feature_data = raw_data_list[i];
        for ( var j = 0; j < feature_data.length; j++ ){
            variable.get_cell(j).set_feature(feature_data[j]);
        }
    }
};

var label_handler = function(data){
    LabelData = data;
    let raw_data_list = [data.pred_train, data.pred_valid, data.pred_test];
    let data_variables_list = [Loader.TrainData, Loader.ValidData, Loader.TestData];
    for( let i = 0; i < raw_data_list.length; i++ ){
        let variable = data_variables_list[i];
        let label_data = raw_data_list[i];
        for ( let j = 0; j < label_data.length; j++ ){
            variable.get_cell(j).set_pred_y(label_data[j]);
        }
    }
    for(let j = 0; j < data.y_train.length; j++){
        Loader.TrainData.get_cell(j).set_y(data.y_train[j]);
    }
};

var sample_handler = function(data){
    SampleData = data;
    var train_sample = data.train;
    var test_sample = data.test;
    var all_sample = data.all;
    var sampling_len = 30,
        width = 1 / sampling_len;
    for( let i = 0; i < train_sample.length; i++ ){
        let d = train_sample[i];
        let cell = Loader.TrainData.get_cell_by_id(d.id);
        cell.add_sample_type(1);
        cell.set_grid_x(d.pos);
        cell.set_width(width);
    }
    for( let i = 0; i < test_sample.length; i++ ){
        let d = test_sample[i];
        let cell  = Loader.TestData.get_cell_by_id(d.id);
        cell.add_sample_type(2);
        cell.set_grid_x(d.pos);
        cell.set_width(width);

    }
    for( let i = 0; i < all_sample.length; i++ ){
        let d = all_sample[i];
        let cell = Loader.TrainData.get_cell_by_id(d.id) || Loader.TestData.get_cell_by_id(d.id);
        cell.add_sample_type(4);
        cell.set_grid_x(cell.get_grid_x().concat(d.pos));
        cell.set_width(width);
    }
    // NavigationView.initialize_nav_data(data);
};

var grid_handler = function(data){
    console.log(data);
    var current_lens = LensView;
    GridLayoutData = data.layout;
    var grid_size = Math.ceil(Math.sqrt(GridLayoutData.length)),
        width = 1 / grid_size;
    GridLayoutData.forEach(d => {
        let cell = null;
        if (LensView.get_data_type()[0] === "train") {
            cell = Loader.TrainData.get_cell_by_id(d.id);
            console.log("get training cell");
        } else if (LensView.get_data_type()[0] === "test") {
            cell = Loader.TestData.get_cell_by_id(d.id);
            console.log("get test cell", cell);
        } else {
            cell = Loader.TrainData.get_cell_by_id(d.id) || Loader.TestData.get_cell_by_id(d.id);
        }
        d.cell = cell;
        var x = Math.round(d.pos[0] * grid_size),
            y = Math.round(d.pos[1] * grid_size);
        d.coord = [x, y];
    });
    var grids = GridLayoutData.map(d => Object({
        coord: d.coord,
        grid_x: d.pos,
        width: width,
        ...d.cell
    }));
    var boundary = data.boundary;

    current_lens.update_grid(data.id, grids, boundary, data.distribution);

    d3.select("#preloader").style("display", "none");
};

// var boundary_handler = function(current_lens, data) {
//     Loader.BoundaryData = data;
//     current_lens.update_boundary_lens();
// };

var entropy_handler = function(data) {
    Loader.EntropyData = data;
    console.log(data);
    var entropy_list = [data.train_entropy, data.test_entropy];
    var data_variables_list = [Loader.TrainData, Loader.TestData];
    for( var i = 0; i < entropy_list.length; i++){
        var variable = data_variables_list[i];
        var entropy_data = entropy_list[i];
        for (var j = 0; j < entropy_data.length; j++){
            variable.get_cell(j).set_entropy(entropy_data[j]);
        }
    }
};

var prediction_handler = function(data){
    Loader.PredictionData = data;
    var prediction_list = [data.train_pred_y, data.test_pred_y];
    var data_variables_list = [Loader.TrainData, Loader.TestData];
    for( var i = 0; i < prediction_list.length; i++ ){
        var variable = data_variables_list[i];
        var prediction_data = prediction_list[i];
        for( var j = 0; j < prediction_data.length; j++){
            variable.get_cell(j).set_pred_y(prediction_data[j]);
        }
    }
};


var focus_handler = function(current_lens, data) {
    let infos = data.info;
    // add_multifocus_lens(current_lens, data);
    InfoView.update_neighbours(data.similar_instances);
};