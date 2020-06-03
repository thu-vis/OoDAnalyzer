function expandAll(){
  var elem = document.querySelector("#cp-container");
  var instance = M.Collapsible.init(elem,
      {accordion:false});
  for(var i = 0; i < 5; i++){
    instance.open(i);
  }
}



var entropy_slider = document.getElementById("entropy-slider");
noUiSlider.create(entropy_slider, {
    start: [0, 0.4, 0.6, 1],
    connect: true,
    step: 0.1,
    orientation: "horizontal",
    range:{
        'min': [0],
        'max': [1.0]
    }
});
entropy_slider.noUiSlider.on("change", entropy_threshold_update);


var neighbour_num_slider = document.getElementById("neighbour-num-slider");
noUiSlider.create(neighbour_num_slider, {
    start: [5],
    step: 1,
    orientation: "horizontal",
    range:{
        'min': [3],
        'max': [9]
    },
    format: wNumb({
        decimals: 0
   })
});
neighbour_num_slider.noUiSlider.on("change", neighbour_num_update);


d3.selectAll(".noUi-connect")
    .each(function(d, i) {
        d3.select(this)
            .style("background", CategorySequentialColor[2][i]);
    });
d3.selectAll(".noUi-origin")
    .each(function(d, i) {
        if (i === 0 || i === 3) {
            d3.select(this).style("display", "none");
        }
        d3.select(this).style("")
    });



$(document).ready(function(){
    $('select').formSelect();
});
$("#dataset-select").change(dataset_selection);

expandAll();

$(window)
    .keydown(function(event) {
        if (event.key === "Control") {
            ControlPressed = true;
            d3.selectAll("image").style("pointer-events", "none");
        }
    })
    .keyup(function(event) {
        if (event.key === "Control") {
            ControlPressed = false;
            d3.selectAll("image").style("pointer-events", "auto");
        }
    });

const d_rollback="M793 242H366v-74c0-6.7-7.7-10.4-12.9-6.3l-142 112c-4.1 3.2-4.1 9.4 0 12.6l142 112c5.2 4.1 12.9 0.4 12.9-6.3v-74h415v470H175c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h618c35.3 0 64-28.7 64-64V306c0-35.3-28.7-64-64-64z";
const d_scan="M136 384h56c4.4 0 8-3.6 8-8V200h176c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H196c-37.6 0-68 30.4-68 68v180c0 4.4 3.6 8 8 8zM648 200h176v176c0 4.4 3.6 8 8 8h56c4.4 0 8-3.6 8-8V196c0-37.6-30.4-68-68-68H648c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8zM376 824H200V648c0-4.4-3.6-8-8-8h-56c-4.4 0-8 3.6-8 8v180c0 37.6 30.4 68 68 68h180c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zM888 640h-56c-4.4 0-8 3.6-8 8v176H648c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h180c37.6 0 68-30.4 68-68V648c0-4.4-3.6-8-8-8zM904 476H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8z";
const d_select="M880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h360c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H184V184h656v320c0 4.4 3.6 8 8 8h56c4.4 0 8-3.6 8-8V144c0-17.7-14.3-32-32-32zM653.3 599.4l52.2-52.2c4.7-4.7 1.9-12.8-4.7-13.6l-179.4-21c-5.1-0.6-9.5 3.7-8.9 8.9l21 179.4c0.8 6.6 8.9 9.4 13.6 4.7l52.4-52.4 256.2 256.2c3.1 3.1 8.2 3.1 11.3 0l42.4-42.4c3.1-3.1 3.1-8.2 0-11.3L653.3 599.4z";

d3.select("#cropping").on('click', function() {
    var mode = d3.select(this).select("path").attr("d") === d_scan ? "exploring" : "cropping";
    if (mode === "exploring") {
        set_cropping();
    } else {
        set_exploring();
    }
});

d3.select("#selecting").on('click', function() {
    var mode = d3.select(this).select("path").attr("d") === d_select ? "exploring" : "selecting";
    if (mode === "exploring") {
        set_selecting();
    } else {
        set_exploring();
    }
});

d3.select("#select-classes").on("click", function() {
    if (d3.select(this).classed("grey")) {
        d3.select(this).classed("grey", false);
        d3.select("#reload-layout").classed("grey", true);
        d3.selectAll("rect.legend").each(function() {
            d3.select(this).style("fill", d3.select(this).style("stroke"));
        });
    } else {
        d3.select(this).classed("grey", true);
        d3.select("#reload-layout").classed("grey", false);
        d3.selectAll("rect.legend").style("fill", "white");
    }
});

d3.select("#reload-layout").on("click", function() {
    if (d3.select(this).classed("grey")) {
    } else {
        var category_shown = Array(LabelNames.length).fill(false),
            total_selected_categories = 0;
        d3.selectAll("rect.legend").each(function(d, i) {
            if (d3.select(this).style("fill") !== "white") {
                category_shown[i] = true;
                total_selected_categories += 1;
            }
        });
        if (total_selected_categories < 2) {
            alert("Please select more than 2 categories");
            d3.selectAll("rect.legend").each(function() {
                d3.select(this).style("fill", d3.select(this).style("stroke"));
            });
        } else {
            LabelShown = category_shown;
            NavigationView.reset();
            $(".display").prop("checked", false).removeClass("with-gap");
            $(".display#test").prop("checked", true);
            $(".comparison").prop("checked", false);
            LensView.switch_datatype("test");
        }
        d3.select(this).classed("grey", true);
        d3.select("#select-classes").classed("grey", false);
    }
});
