/**
 * Created by Changjian on 2018/12/17.
 */

var legend_create = function() {
    var legend_y_shift = 10;
    var legend_height = 20;
    var legend_width = 20;
    var legend_padding = 10;
    var text_height = legend_height;
    var text_x_shift = legend_width + 10;
    var text_y_shift = legend_y_shift;
    var text_width = 40;
    var text_padding = legend_padding;
    var text_px = 15;
    var legend_group = d3.select("#legend-group");
    legend_group.selectAll("*").remove();
    var bbox = legend_group.node().getBoundingClientRect();
    legend_group.attr("width", bbox.width)
        .attr("height", Math.ceil(LabelNames.length / 3) * (legend_height + legend_padding) + legend_padding)
        .attr("transform", "translate(" + ( 0 ) + "," + legend_padding + ")");
    // classes
    legend_group.selectAll("rect.legend")
        .data(LabelNames)
        .enter()
        .append("rect")
        .attr("class", "legend");
    legend_group.selectAll("text.legend")
        .data(LabelNames)
        .enter()
        .append("text")
        .attr("class", "legend")
        .style("fill", "#333");
    legend_group.selectAll("rect.legend")
        .data(LabelNames)
        .attr("width", legend_width)
        .attr("height", legend_height)
        .attr("x", (d, i) => (i % 3) * (legend_width + text_x_shift + text_width))
        .attr("y", (d, i) => (i - i % 3) / 3 * (legend_height + legend_padding))
        .style("stroke", (d, i) => CategoryColor[i])
        .style("stroke-width", 1)
        .style("fill", (d, i) => CategoryColor[i])
        .style("cursor", "pointer")
        .on("click", function(d, i) {
            if (!d3.select("#select-classes").classed("grey")) {
                return;
            }
            if (d3.select(this).style("fill") !== "white") {
                d3.select(this).style("fill", "white");
            } else {
                d3.select(this).style("fill", d3.select(this).style("stroke"));
            }
        });
    legend_group.selectAll("text.legend")
        .data(LabelNames)
        .attr("x", (d, i) => (i % 3) * (legend_width + text_x_shift + text_width) + text_x_shift)
        .attr("y", (d, i) => (i - i % 3) / 3 * (legend_height + legend_padding) + text_height / 2 + text_px / 2)
        .attr("font-size", text_px + "px")
        .attr("text-anchor", "start")
        .text(d => d);

    // // types
    // legend_group.append("rect")
    //     .attr("width", 20)
    //     .attr("height", 20)
    //     .attr("x", 100)
    //     .attr("y", 0)
    //     .style("fill", "grey");
    // legend_group.append("text")
    //     .attr("x", text_x_shift + 80)
    //     .attr("y", text_height / 2 + text_px / 2)
    //     .attr("font-size", text_px + "px")
    //     .attr("text-anchor", "start")
    //     .text("training");
    // legend_group.append("circle")
    //     .attr("r", 10)
    //     .attr("cx", 10 + 100)
    //     .attr("cy", 10 + (legend_height + legend_padding))
    //     .style("fill", "grey");
    // legend_group.append("text")
    //     .attr("x", text_x_shift + 80)
    //     .attr("y", text_height + text_padding + text_height / 2 + text_px / 2)
    //     .attr("font-size", text_px + "px")
    //     .attr("text-anchor", "start")
    //     .text("test");
};

var on_plot_clicked = function(d, i){
    console.log("clicked");
    var img_url = ImageApi + "?dataset=" + DatasetName + "&filename=" + i + ".jpg";
    InfoView.update_image(img_url);
};


var change_projection_method = function(embed_method){
    if (!embed_method){
        console.log("no embed method is provided.")
    }
    var url = GridLayoutApi + "?dataset=" + DatasetName + "&embed-method=" + embed_method;
    var node = new request_node(url, embed_data_handler, "json", "GET");
    node.notify();
    LensView.update_distribution_lens(null);
};

var info_panel_focus = function(id){
    infos = {

    };
    InfoView.update_info(infos);
};


/*
* triggered functions in control panel
* */
var dataset_selection = function(){
    let selection = $("#dataset-select").find("option:selected").text();
    console.log("present dataset selection: ", selection);
    remove_dom();
    DatasetName = selection;
    set_up(DatasetName);
    load_data();
};

var display_selection = function(display_type){
    if (display_type !== "train" && display_type !== "test") {
        if (!(LensView.get_position_mode() === display_type ||
            (LensView.get_position_mode() === "superposition" && display_type === "all"))) {
            $(".display").prop("checked", true).addClass("with-gap");
            $(".comparison").prop("checked", false);
            $("#" + display_type).prop("checked", true);
            LensView.switch_datatype(display_type);
        }
    } else {
        if (LensView.get_data_type() !== display_type) {
            $(".display").prop("checked", false).removeClass("with-gap");
            $(".display#" + display_type).prop("checked", true);
            $(".comparison").prop("checked", false);
            LensView.switch_datatype(display_type);
        }
    }
};




var position_selection = function(){
    var name = document.getElementsByName("group-position");
    var selection = null;
    for (var i = 0; i < name.length; i++){
        if (name[i].checked){
            selection = name[i].id;
            break;
        }
    }
    console.log("present filter selection: ", selection);
};

var switch_boundary = function() {
    var name = document.getElementById("boundary-switch");
    LensView.switch_boundary_lens(name.checked);
};

var switch_images = function() {
    var name = document.getElementById("images-switch");
    LensView.switch_images(name.checked);
};

var switch_entropy = function() {
    var name = document.getElementById("entropy-switch");
    LensView.switch_entropy_lens(name.checked);
};

var switch_info = function() {
    var name = document.getElementById("info-switch");
    if (name.checked) {
        d3.select("#info-oi").classed("text-black", false);
        d3.select("#info-sm").classed("text-black", true);
    } else {
        d3.select("#info-oi").classed("text-black", true);
        d3.select("#info-sm").classed("text-black", false);
    }
    InfoView.switch_display_images(name.checked);
};

var neighbour_num_update = function(values, handle, unencoded, tap, positions) {
    console.log(values, handle, unencoded, tap, positions);
    InfoView.update_neighbour_num(unencoded);
};

var entropy_threshold_update = function(values, handle, unencoded, tap, positions){
    console.log(values, handle, unencoded, tap, positions);
    LensView.update_entropy_threshold(unencoded);
};


// var withdraw_from_compare = function() {
//     LensView.remove_compare();
// };

var set_cropping = function() {
    LensView.set_mode("cropping");
};

var set_exploring = function() {
    LensView.set_mode("exploring");
};

var set_selecting = function () {
    LensView.set_mode("selecting");
};

var tips_in_another = function(f, state, type){
    LensView.DistributionLens.tips_in_another(f, state, type);
};