/**
 * Created by Changjian on 2018/12/17.
 */

/*
    Notice: variables like AbcDefgHijk are globals ones
            variables like abc_defg_hijk are local ones
 */


var load_data = function () {
    /*
    * load data that need to be stored in global variables
    * */
    console.log("loading data...");

    Loader.manifest_node.notify();
};


var set_up = function(dataset){
    // DisView = new DistributionLayout(d3.select("#block-2-1"));
    Loader = new DataLoader(dataset);
    InfoView = new InfoLayout(d3.select("#info-panel"));
    LensView = new LensLayout(d3.select("#block-2-1"));
    NavigationView = new NavigationTree(d3.select("#block-1-2"));
    LensView.set_navigation(NavigationView);
    // SnapshotView = new Snapshot(d3.select("#block-3-2"));
};

var remove_dom = function(){
    // d3.select("#block-1-1").selectAll("svg").remove();
    d3.select("#block-1-2").selectAll("svg").remove();
    d3.select("#block-2-1").selectAll("svg#lens-svg").remove();
    d3.select("#block-info").selectAll("svg").remove();
    d3.select("#block-neighbour").selectAll("svg").remove();
    // d3.select("#block-3-2").selectAll("svg").remove();

    $(".comparison").prop("checked", false);
    $(".display").prop("checked", false).prop("disabled", false);
    $(".display#test").prop("checked", true);
    LensView.switch_datatype("test");
};

$(document).ready(function () {
    document.oncontextmenu = () => false;

    // DatasetName = "Dog-Cat";
    DatasetName = "Animals";
    // DatasetName = "Animals-leopard";

    set_up(DatasetName);
    load_data();
});
