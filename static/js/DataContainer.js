/**
 * Created by Changjian on 2018/12/17.
 * 
 * Definition of DataContainer, a data structure to store data needed for visualization
 */

var DataCell = function(_id, _img_url_prefix, _thumbnail_url_prefix ,_data_type){
    var that = this;
    var id = _id === null ? -1 : _id;
    var embed_x = [-0.1, -0.1];
    var feature = null;
    var y = null;
    var pred_y = null;
    var grid_x = [-0.1, -0.1];
    var coord = [-1, -1];
    var img_width = null;
    var img_url_prefix = _img_url_prefix;
    var thumbnail_url_prefix = _thumbnail_url_prefix;
    var entropy = null;
    var confidence = null;
    var data_type = _data_type;
    var visible = false;
    var sample_type = 0;
    var datatype = "";

    this.get_id = function(){
        return id;
    };
    this.get_img_url = function(){
        return img_url_prefix + "&filename=" + id + ".jpg";
    };
    this.get_thumbnail_url = function(){
        return thumbnail_url_prefix + "&filename=" + id + ".jpg";
    };
    this.get_saliency_map_url = function(){
        return SaliencyMapApi + "?dataset=" + DatasetName + "&filename=" + id + ".jpg";
    };

    this.set_embed_x = function(x){
        embed_x = x;
    };
    this.get_embed_x = function(){
        return embed_x;
    };
    this.set_feature = function(f){
        feature = f;
    };
    this.get_feature = function(){
        return feature;
    };
    this.set_y = function(_y){
        y = _y;
    };
    this.get_y = function(){
        if (data_type === "test"){
            return pred_y;
        }
        else if (data_type === "train"){
            return y;
        }
        return y;
    };
    this.set_pred_y = function(y){
        pred_y = y;
    };
    this.get_pred_y = function(){
        return pred_y;
    };
    this.set_confidence = function(_confidence) {
        confidence = _confidence;
    };
    this.get_confidence = function(){
        return confidence;
    };
    this.set_grid_x = function(x){
        grid_x = x;
    };
    this.get_grid_x = function(){
        return grid_x;
    };
    this.set_coord = function(_coord){
        coord = _coord;
    };
    this.get_coord = function(){
        return coord;
    };
    this.set_width = function(x){
        img_width = x;
    };
    this.get_width = function(){
        return img_width;
    };
    this.set_entropy = function(x){
        entropy = x;
    };
    this.get_entropy = function(){
        return entropy;
    };

    this.set_visible = function(state){
        visible = state;
    };
    this.get_visible = function(){
        return visible;
    };
    this.add_sample_type = function(type){
        sample_type = sample_type + type;
    };
    this.get_sample_type = function(){
        // transform the return value in to binary number
        // 1st bit: training
        // 2nd bit: test
        // 3rd bit: all
        return sample_type;
    };
    this.set_datatype = function(_type) {
        datatype = _type;
    };
    this.get_datatype = function() {
        return datatype;
    };
};

var DataContainer = function(_data_type, _img_url_prefix, _thumbnail_url_prefix){
    let that = this;
    let data_type = _data_type;
    let datas = [];
    let img_url_prefix = _img_url_prefix;
    let id_to_idx_map = {};

    this.get_img_url_prefix = function(){
        return img_url_prefix;
    };

    this.get_thumbnail_url_prefix = function(){
        return _thumbnail_url_prefix;
    };

    this.append_datacell = function(cell){
        datas.push(cell);
        id_to_idx_map[cell.get_id()] = datas.length - 1;
    };

    this.get_all_embed_x = function(){
       var embed_xs = [];
       for (var i = 0; i < datas.length; i++){
           embed_xs.push(datas[i].get_embed_x())
       }
       return embed_xs;
    };

    this.get_all_y = function(){
        var all_y = [];
        for ( var i = 0; i < datas.length; i++ ){
            all_y.push(datas[i].get_y())
        }
        return all_y;
    };

    this.get_data_type = function(){
        return data_type;
    };

    this.get_cell = function(idx){
        return datas[idx];
    };

    this.get_cell_by_id = function(id){
        var cell_pos = id_to_idx_map[id];
        if (cell_pos !== undefined) {
            return datas[id_to_idx_map[id]];
        } else {
            return null;
        }
    };

    this.get_all_data = function(){
        return datas;
    };

    this.unset_all_visible_state = function(){
        for (let i = 0; i < datas.length; i++ ){
            datas[i].set_visible(false);
        }
    };

    this.export = function(){
    //TODO: this function is designed for data manipulation
    };
};