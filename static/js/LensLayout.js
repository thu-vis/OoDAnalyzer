var LensLayout = function(container) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height;

    var datatype = "test"; // Default = test
    var display_type = "single";

    var datacells = Array();

    var svg = that.container.append("svg");
    var another = null;
    that.nav = null;

    const NUM_OF_LENS = 10;
    var lens_status = Array(NUM_OF_LENS).fill(-1); // -1 for hidden, others represents z-index in lens

    that.DistributionLens = null;
    that.EntropyLens = null;
    var lens = Array(NUM_OF_LENS).fill(null);
    // var mouse_pressed = false;

    function zoomed(x, y, k) {
        that.DistributionLens.zoomed(x, y, k);
    }

    that.zoom = function(transform) {
        d3.zoom().transform(svg, transform);
        const {x, y, k} = transform;
        zoomed(x, y, k);
    };

    that._init = function(){
        // set up svg's width and height
        svg.attr("width", layout_width)
            .attr("height", layout_height)
            .attr("id", "lens-svg");
            // .style("float", "left")
            // .style("padding-left", bbox.width * 0.01);
        lens[0] = that.DistributionLens = new DistributionLayout(svg, that);
        lens[5] = that.EntropyLens = new Entropy(svg, that);
        // svg.call(d3.zoom().scaleExtent([1/2, 16]).on("zoom", () => {
        //     const {x, y, k} = d3.event.transform;
        //     zoomed(x, y, k);
        //     if (another != null) {
        //         another.zoom(d3.event.transform);
        //     }
        //     // that.set_viewbox(d3.event.transform);
        // }));
        svg.on("dblclick.zoom", null);
        lens_status[0] = 0;
        lens_status[1] = -1;
        lens_status[2] = -1;
        lens_status[3] = -1;
        lens_status[4] = -1;
        lens_status[5] = 5;

    };
    that.set_navigation = function(nav) {
        that.nav = nav;
    };

    that.init = function(){
        console.log("lenslayout init");
        that._init();
    }.call();

    that.destroy = function() {
        svg.remove();
    };

    that.set_mode = function(_mode){
        mode = _mode;
        if (_mode === "cropping") {
            d3.select("#cropping").select("path").attr("d", d_rollback);
            d3.select("#selecting").select("path").attr("d", d_select);
            that.DistributionLens.enter_overview();
        } else if (_mode === "selecting") {
            d3.select("#selecting").select("path").attr("d", d_rollback);
            d3.select("#cropping").select("path").attr("d", d_scan);
            that.DistributionLens.enter_overview();
        } else if (_mode === "exploring") {
            d3.select("#cropping").select("path").attr("d", d_scan);
            d3.select("#selecting").select("path").attr("d", d_select);
            that.DistributionLens.quit_overview();
        }
    };
    that.get_mode = function() {
        return mode;
    };

    that.get_position_mode = function() {
        return display_type;
    };
    that.get_data = function() {
        return datacells;
    };

    that.draw = function(){
        that.create();
        // that.update();
        that.remove();
    };
    that.create = function(){
    };
    that.update = function(){
        var lens_status_sort_index = [...Array(NUM_OF_LENS).keys()]
            .sort((x, y) => lens_status[x] === lens_status[y] ? 0 : lens_status[x] > lens_status[y] ? 1 : -1);
        for (let idx of lens_status_sort_index) {
            if (lens_status[idx] < 0)
                continue;
            lens[idx].draw()
        }
    };
    that.remove = function(){
    };

    that.update_entropy_threshold = function(value) {
        that.EntropyLens.update_threshold(value);
        that.update_entropy_lens();
    };


    that.refresh = function() {
        var nav_status = that.nav.switch_datatype(datatype);
        if (nav_status !== null) {
            that.update_grid(nav_status.id, nav_status.node.display_data, nav_status.node.boundary_data);
        } else {
            var category_code = LabelShown.reduce((str, d) => str + (d ? "1" :"0"), "");
            d3.select("#preloader").style("display", "block");
            Loader.grid_data_node.set_url(
                GridLayoutApi + "?dataset=" + Loader.dataset + "&datatype=" + datatype + "&embed-method=tsne" +
                "&left-x=" + (0) +
                "&top-y=" + (0) +
                "&width=" + (1) +
                "&height=" + (1) +
                "&distribution=" + datatype +
                "&node-id=" + (-1) +
                "&class=" + category_code
            );
            Loader.grid_data_node.set_on();
        }
    };
    that.get_data_type = function() {
        return datatype;
    };

    // Interactions
    that.switch_datatype = function(_datatype) {
        // if (_datatype !== "juxtaposition") {
        //     if (display_type !== "juxtaposition") {
        //         if (datatype === _datatype) {
        //             return;
        //         } else {
        //             datatype = _datatype;
        //             that.refresh();
        //         }
        //     } else {
        //          that.remove_compare();
        //          datatype = _datatype;
        //          that.refresh();
        //     }
        //     if (_datatype === "all") {
        //         display_type = "superposition";
        //     } else {
        //         display_type = "single";
        //     }
        // } else {
        //     that.compare();
        //     display_type = "juxtaposition";
        // }
        if (_datatype === "juxtaposition") {
            that.juxtaposition();
        } else {
            if (display_type === "juxtaposition") {
                that.nav.merge();
            }
            if (_datatype === "all") {
                that.superpostion();
            } else {
                display_type = "single";
                datatype = _datatype;
                that.refresh();
            }
        }
    };
    that.switch_lens = function(_len_name){
        if(_len_name === "scatter-plot"){
            lens_status[0] = -1;
        }
        else if(_len_name === "grid-layout"){
            lens_status[0] = 0;
        }
        else{
            console.log("unsupported len type");
        }

        that.update_layout();
    };
    that.switch_filter = function(_filter) {
        if (_filter === "none-filter") {
            lens_status[4] = -1;
            lens_status[5] = -1;
        } else if (_filter === "entropy-filter") {
            lens_status[4] = -1;
            lens_status[5] = 5;
        } else if (_filter === "confidence-filter") {
            lens_status[4] = 4;
            lens_status[5] = -1;
        }
        that.update_filter();
    };

    that.update_distribution_lens = function(id, data, boundary_points) {
        that.DistributionLens.load_data(id, data, boundary_points);
        that.DistributionLens.draw();
    };


    that.update_entropy_lens = function() {
        if (lens_status[5] > -1) {
            that.EntropyLens.draw();
        } else {
            that.EntropyLens.remove();
        }
    };



    that.switch_images = function(visible) {
        console.log("switch images: ", visible);
        if (visible){
            that.DistributionLens.setImageVisible("visible");
        }
        else {
            that.DistributionLens.setImageVisible("hidden");
        }
    };
    that.switch_entropy_lens = function(visible) {
        lens_status[5] = visible ? 5 : -1;
        that.update_entropy_lens();
    };


    that.resize = function() {
        layout_width = bbox.width * 0.5;
        layout_height = bbox.height - 28;
        svg.attr("width", layout_width)
            .attr("height", layout_height);
        zoomed(0, layout_height / 4, 0.75);
        const transform = d3.zoomTransform(0).translate(0, 255).scale(0.75);
        d3.zoom().transform(svg, transform);
        d3.select("#snapshot-or-return")
            .on('click', function() {
                withdraw_from_compare();
            })
            .select("i")
            .html("keyboard_backspace");
    };
    that.compare = function(snapshot) {
        another = new LensLayout(that.container);
        another.set_another(that);
        NavigationView.split(another);
        that.switch_datatype("train");
        another.switch_datatype("test");
        // another.sync_status();
        // Re-Layout
        that.resize();
        another.resize();
    };
    that.remove_compare = function() {
        another.destroy();
        another = null;
        NavigationView.merge();
        layout_width = bbox.width;
        layout_height = bbox.height - 28;
        svg.attr("width", layout_width)
            .attr("height", layout_height);
        zoomed(0, 0, 1);
        const transform = d3.zoomTransform(0).translate(0, 0).scale(1);
        d3.zoom().transform(svg, transform);
    };
    that.get_another = function() {
        return another;
    };
    that.set_another = function(_another) {
        another = _another;
    };
    that.tips_in_another = function(f, tip_type) {
        var dist = function(x, y) {
            var len = x.length;
            var sum = 0;
            for (var i = 0; i < len; ++i) {
                sum += (x[i] - y[i])**2;
            }
            return Math.sqrt(sum);
        };

        if (!another) {
            return;
        } else {
            var distances = another.get_data().map(d => Object({
                dist: dist(d.get_feature(), f),
                id: d.get_id()
            }));
            // var pos = distances.indexOf(Math.min(...distances));
            // var id = another.get_data()[pos].get_id();
            distances.sort((x, y) => x.dist - y.dist);
            var k = Math.min(8, distances.length);
            if (tip_type) {
                for (let i = 0; i < k; ++i) {
                    another.highlight_grid(distances[i].id);
                }
            } else {
                for (let i = 0; i < k; ++i) {
                    another.dehighlight_grid(distances[i].id);
                }
            }
        }
    };

    that.highlight_grid = function(id) {
        svg.select("#ID-" + id).select("rect")
            .style("stroke-width", 2.0);
    };

    that.dehighlight_grid = function(id) {
        svg.select("#ID-" + id).select("rect")
            .style("stroke-width", 0.0);
    };
    that.get_lens_status = function() {
        return lens_status;
    };
    that.sync_status = function() {
        lens_status = another.get_lens_status();
    };
    that.update_grid = function(id, grids, boundary, dis) {
        datacells = grids;
        that.update_distribution_lens(id, datacells, boundary);
        // that.update_recommendation_lens();
        // that.update_boundary_lens(boundary);
        // that.update_multifocus_lens();
        that.update_filter();
        that.save_status(id, grids, boundary, dis);
    };
    that.update_filter = function() {
        // remove filters
        // svg.selectAll("rect.boundingbox").style("fill-opacity", 0.5);
        // update filters
        // that.update_confidence_lens();
        that.update_entropy_lens();
    };
    that.update_layout = function() {
    };

    // Lens status Query
    that.boundary_lens_isVisible = function() {
        return lens_status[1] !== -1;
    };
    that.recommendation_lens_isVisible = function() {
        return lens_status[2] !== -1;
    };

    // Navigation
    that.save_status = function(id, grids, boundary, dis) {
        var status = Object();
        status.id = id;
        status.datatype = dis;
        status.sampling_area = that.DistributionLens.get_sampling_area();
        status.display_data = grids;
        status.boundary_data = boundary;
        that.nav.load_data(id, status);
        that.nav.draw();
    };
    that.load_status = function(id, status) {
        that.update_grid(status.id, status.display_data, status.boundary_data);
        that.DistributionLens.set_sampling_area(status.sampling_area);
    };

    // Switch position mode
    that.juxtaposition = function() {
        display_type = "juxtaposition";
        datatype = "both";
        that.nav.split();
        let train_nav = that.nav.get_root("train"),
            test_nav = that.nav.get_root("test");
        let flag = "";
        if (train_nav !== null) {
            that.DistributionLens.load_data(train_nav.id, train_nav.node.display_data, train_nav.node.boundary_data);
        } else {
            that.nav.switch_datatype("train");
            flag = "train";
        }
        if (test_nav !== null) {
            that.DistributionLens.load_data(test_nav.id, test_nav.node.display_data, test_nav.node.boundary_data);
        } else {
            that.nav.switch_datatype("test");
            flag = "test";
        }
        if (!flag) {
            that.DistributionLens.draw();
            that.update_filter();
            that.nav.draw();
        } else {
            var category_code = LabelShown.reduce((str, d) => str + (d ? "1" :"0"), "");
            d3.select("#preloader").style("display", "block");
            Loader.grid_data_node.set_url(
                GridLayoutApi + "?dataset=" + Loader.dataset + "&datatype=" + flag  + "&embed-method=tsne" +
                "&left-x=" + (0) +
                "&top-y=" + (0) +
                "&width=" + (1) +
                "&height=" + (1) +
                "&distribution=" + flag +
                "&node-id=" + (-1) +
                "&class=" + category_code
            );
            Loader.grid_data_node.set_on();
        }
    };

    that.superpostion = function() {
        display_type = "superposition";
        datatype = "all";
        that.refresh();
    };

};