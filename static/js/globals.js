/**
 * Created by Changjian on 2018/12/17.
 */


/*
    System information
*/
var ManifestApi = "/api/manifest";
var DataApi = "/api/embed-data";
var IdxApi = "/api/idx";
var FeatureApi = "/api/feature";
var LabelApi = "/api/label";
var SampleApi = "/api/original-samples";
var ImageApi = "/api/image";
var ThumbnailApi = "/api/thumbnail";
var SaliencyMapApi = "/api/saliency-map";
var GridLayoutApi = "/api/grid-layout";
var BoundaryApi = "/api/decision-boundary";
var EntropyApi = "/api/entropy";
var PredictionApi = "/api/prediction";
var ConfidenceApi = "/api/confidence";
var FocusApi = "/api/focus/";

/*
*  View Object
* */
var ControlView = null;
var DisView = null;
var InfoView = null;
var LensView = null;
var SnapshotView = null;
var NavigationView = null;

/*
    Const variables for data storage
*/
var DatasetName = null;
var TrainInstanceNum = null;
var ValidInstanceNum = null;
var TestInstanceNum = null;
var FeatureNum = null;
var LabelNames = null;
var LabelShown = null;
var TrainAcc = null;
var TrainMinWidth = null;
var TestMinWidth = null;
var DataType = null;

/*
*  intermediate variables for data storage
* */
// var TrainData = null;
// var ValidData = null;
// var TestData = null;

/*
*  Color
* */
var CategoryColor = [
    "#4fa7ff",
    "#ffa953",
    "#55ff99",
    "#ba9b96",
    "#c982ce",
    "#bcbd22",
    "#e377c2",
    "#990099",
    "#17becf",
    "#8c564b"
];

// 蓝色
    // ["#deebf7",
    // "#60a9ce",
    // "#225982"],

// 橙色
//     ["#fef2e6",
//     "#fd9036",
//     "#f36c29"]

    // ["#dfefff",
    // "#4fa7ff",
    // "#0063c6"], //蓝色
    //
    // ["#ffe5cc",
    // "#ffa953",
    // "#cc6600"], // 橙色； shixia


var CategorySequentialColor = [
    ["#c8e3ff",
    "#4fa7ff",
    "#0063c6"], //蓝色

    ["#ffe5cc",
    "#ffa953",
    "#cc6600"], //

    ["#bfffd9",
    "#22ff7a",
    "#00993e"], //

    ["#e6dbd9",
    "#ba9b96",
    "#7a5650"], //

    ["#ecd3ed",
    "#c982ce",
    "#8b3892"], //
];

var ThemeColor = "#26a69a";
var Gray = "#a8a8a8";

/*
    variables that debug needed
*/
var AnimationDuration = 1000;
var ManifestData = null;
var EmbedData = null;
var IdxData = null;
var LabelData = null;
var SampleData = null;
var GridLayoutData = null;
var BoundaryData = null;

/*
    variables that needs careful usage
*/
var TrueTrainLabels = null;
var TrueValidLabels = null;
var TrueTestLabels = null;

var Loader = null;

/*
    Keyboard status
 */
var ControlPressed = false;