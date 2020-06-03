var Confidence = function(container, parent) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var plot_width = layout_width;
    var plot_height = layout_height - 20;
    plot_width = plot_height;

    var threshold = 0.5;

    var grids = null;

    that._init = function(){
        console.log("confidence init");
    };


    that.init = function(){
        that._init();
    }.call();

    that.draw = function(){
        that.update();
    };


    that.update = function() {
        grids = container.selectAll("rect.boundingbox");
        var data = parent.get_data();
        // grids.each(function(d, i) {
        //     d3.select(this).style("fill-opacity", function(){
        //         // return Math.pow(d.get_entropy(), 0.4) + 0.2;
        //         var conf = d.get_confidence();
        //         if (conf < threshold){
        //             return 1;
        //         }
        //         else{
        //             return 0.5;
        //         }
        //     })
        // })
    };

    that.remove = function(){
        // grids.style("fill-opacity", 0.5);
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

};