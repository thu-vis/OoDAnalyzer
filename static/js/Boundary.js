var Boundary = function(container) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var plot_width = layout_width;
    var plot_height = layout_height - 20;
    plot_width = plot_height;


    var keypoints = null;

    var group_container = that.container.append("g").attr("id", "boundary-group").style("visibility", "hidden");
    var seg_group = group_container.append("g").attr("id", "seg-group");
    var boundary = null;

    that._init = function(){
        console.log("boundary init");
        group_container.attr("width", layout_width)
            .attr("height", layout_height);
        seg_group.attr("transform", "translate(10, 10)");
        boundary = seg_group.append("polyline")
            .style("stroke-width", 10)
            .style("stroke", "red")
            .style("fill", "none")
            .style("stroke-linecap", "round")
            .style("storke-linejoin", "round")
            .style("pointer-events", "none")
            .style("opacity", "0.5");
    };

    that.init = function(){
        that._init();
    }.call();

    that.draw = function(){
        that.create();
        that.update();
        that.remove();
        // seg_group.transition()
        //     .duration(1000)
        //     .attr("transform", "translate(" + ( 10 ) + "," + ( 10 ) + ")");
    };

    that.get_data = function() {
        return keypoints;
    };

    that.create = function(){
    };

    that.update = function(){
        boundary.transition()
            .duration(1000)
            .attr("points", keypoints.reduce(
            (total, cur) =>
                total + String(cur[0] * plot_width) + "," + String(cur[1] * plot_height) + " ",
        ""));
    };

    that.remove = function(){
    };

    that.load_data = function(data){
        keypoints = data ? data : [];
    };

    that.zoomed = function(x, y, k){
        let t = d3.zoomIdentity;
        t = t.translate(x, y).scale(k);
        group_container.attr("transform",t);
    };

    that.setVisible = function(visible) {
        group_container.style("visibility", visible ? "visible" : "hidden");
    };

    that.get_boundary_data = function() {
        return keypoints;
    }

};