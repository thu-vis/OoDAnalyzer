var Entropy = function(container, parent) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var plot_width = layout_width;
    var plot_height = layout_height - 20;
    plot_width = plot_height;

    var threshold = [0, 0.4, 0.6, 1.0];
    var grids = null;

    that._init = function(){
        console.log("entropy init");
    };

    that.create = function(){
    };

    that.init = function(){
        that._init();
    }.call();

    that.draw = function(){
        that.update();
    };


    that.update = function(){
        grids = d3.selectAll("g.grid");
        grids.each(function(d) {
            d3.select(this).select("rect").style("fill", function(){
                // return Math.pow(d.get_entropy(), 0.4) + 0.2;
                let ent = d.get_entropy();
                let sequential_color = CategorySequentialColor[d.get_pred_y()];
                if (ent > threshold[2]){
                    return sequential_color[2];
                }
                else if (ent > threshold[1]){
                    return sequential_color[1];
                }
                else{
                    return sequential_color[0];
                }
            })
        })
    };

    that.remove = function(){
        grids = d3.selectAll("g.grid");
        grids.each(function(d) {
            d3.select(this).select("rect").style("fill",  CategorySequentialColor[d.get_pred_y()][0]);
        })
    };

    that.update_threshold = function(value){
        threshold = value;
    };

    that.zoomed = function(){
        const {x, y, k} = d3.event.transform;
        let t = d3.zoomIdentity;
        t = t.translate(x, y).scale(k);
        group_container.attr("transform",t);
    };

    that.setVisible = function(visible) {
    }
};