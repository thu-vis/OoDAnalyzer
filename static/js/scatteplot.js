/**
 * Created by Changjian on 2019/03/07.
 */

var ScatterPlot = function(container, parent){
    let that = this;
    that.container = container;

    let bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var plot_width = layout_width;
    var plot_height = layout_height - 20;
    plot_width = plot_height;
    let solid_point_radius = 5;

    let datatype = "train"; // default = "train", option = "test" or "all"
    let all_data = null;

    let group_container = that.container.append("g").attr("id", "scatter-plot");
    let plot_group = group_container.append("g").attr("id", "tsne");

    that._init = function(){
        group_container.attr("width", layout_width)
            .attr("height", layout_height);

        plot_group.attr("transform",
            "translate(" + (10) + "," + (10) + ")");

    };

    that.init = function(){
        that._init();
    }.call();

    that.create = function(){
        that.plot_create();
    };

    that.update = function(){
        that.plot_update();
    };

    that.remove = function(){
        that.plot_remove();
    };

    that.plot_create = function(){
        plot_group.append("rect")
            .attr("x", -10)
            .attr("y", -10)
            .attr("width", plot_width + 20)
            .attr("height", plot_height + 20)
            .attr("fill", "white")
            .attr("stroke", "white");
        plot_group.selectAll("circle.instance")
            .data(all_data)
            .enter()
            .append("circle")
            .attr("class", "instance")
            .attr("id", d => "ID-" + d.get_id());
        plot_group.selectAll("circle.instance")
            .data(all_data)
            .exit()
            .remove();
    };

    that.plot_update = function(){
        plot_group.selectAll("circle.instance")
            .data(all_data)
            .enter()
            .append("circle")
            .attr("class", "instance")
            .attr("id", d => "ID-" + d.get_id());
        plot_group.selectAll("circle.instance")
            .data(all_data)
            .exit()
            .remove();
        plot_group.selectAll("circle.instance")
            .data(all_data)
            .attr("cx", function(d,i){
                return d.get_embed_x()[0] * plot_width;
            })
            .attr("cy", function(d,i){
                return d.get_embed_x()[1] * plot_height;
            })
            .attr("r", solid_point_radius)
            .style("fill", function(d,i){
                return CategoryColor[d.get_y()];
            });
    };

    that.plot_remove = function(){
        plot_group.selectAll("*")
            .remove();
    };

    that._update_state = function(datatype){
        console.log(datatype);
        if(datatype === "train"){
            all_data = Loader.TrainData.get_all_data();
        }
        else if (datatype === "test"){
            all_data = Loader.TestData.get_all_data();
        }
        else if (datatype === "all"){
            let a = Loader.TrainData.get_all_data();
            let b = Loader.TestData.get_all_data();
            all_data = a.concat(b);
        }
        else {
            console.log("scatter plot type is not supported.");
        }
    };

    that.display = function() {
        that._update_state(parent.get_data_type());
        that.create();
        that.update();
        plot_group.style("opacity", 0)
            .transition()
            .duration(500)
            .style("opacity", 1);
    };

    that.hide = function() {
        plot_group.transition()
            .duration(500)
            .style("opacity", 0)
            .on("end", () => {
                that.remove();
            });

    };

    that.update_state = function() {
        that._update_state(parent.get_data_type());
        that.update();
    };
};
