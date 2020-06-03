var Snapshot = function(container) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var thumbnail_size = 0.9 * layout_width * 0.49;
    var offset = 0.05 * thumbnail_size;
    var boundingbox_width = 3;

    var svg = that.container.append("svg");
    var group_banner = svg.append("g").attr("id", "snapshot-banner").attr("transform", "translate(1, 1)");
    var group_container = svg.append("g").attr("id", "snapshot-group").attr("transform", "translate(1, 21)");
    var train_group = group_container.append("g").attr("id", "snapshot-train-group");
    var test_group = group_container.append("g").attr("id", "snapshot-test-group")
        .attr("transform", "translate(" + (thumbnail_size + offset) + ", 0)");

    // data
    var snapshots = Array();

    that._init = function(){
        // set up svg's width and height
        svg.attr("width", layout_width)
            .attr("height", layout_height)
            .attr("transform", "translate(5, 10)")
            .attr("id", "snapshot-svg")
            .style("padding-left", bbox.width * 0.015);
            // .style("float", "left");
        // svg.append("line")
        //         .attr("x1", 1)
        //         .attr("y1", 0)
        //         .attr("x2", 1)
        //         .attr("y2", layout_height)
        //         .style("stroke-width", 1)
        //         .style("stroke", "grey")
        //         .style("stroke-dasharray", "5, 5");
        var onwheel = function(e) {
            var delta = (e.originalEvent.wheelDelta && (e.originalEvent.wheelDelta > 0 ? 1 : -1))||
              (e.originalEvent.detail && (e.originalEvent.detail > 0 ? -1 : 1));
            console.log(delta);
        };
        $("#snapshop-train-group").on("mousewheel DOMMouseScroll", onwheel);
        $("#snapshop-test-group").on("mousewheel DOMMouseScroll", onwheel);

        // Banner
        // group_banner.append("rect")
        //     .attr("x", 7)
        //     .attr("y", 0)
        //     .attr("width", thumbnail_size * 2 + offset)
        //     .attr("height", 18)
        //     .attr("fill", "none")
        //     .attr("stroke", "grey")
        //     .attr("stroke-width", 1);
        // group_banner.append("text")
        //     .attr("x", thumbnail_size + offset / 2)
        //     .attr("y", 9)
        //     .attr("fill", "grey")
        //     .style("text-anchor", "middle")
        //     .style("dominant-baseline", "middle")
        //     .text("Snapshots");
        group_banner.append("rect")
            .attr("x", 7)
            .attr("y", 0)
            .attr("width", thumbnail_size)
            .attr("height", 18)
            .attr("fill", "none")
            .attr("stroke", "grey")
            .attr("stroke-width", 1);
        group_banner.append("text")
            .attr("x", thumbnail_size / 2)
            .attr("y", 9)
            .attr("fill", "grey")
            .style("text-anchor", "middle")
            .style("dominant-baseline", "middle")
            .text("training");
        group_banner.append("rect")
            .attr("x", thumbnail_size + offset + 7)
            .attr("y", 0)
            .attr("width", thumbnail_size)
            .attr("height", 18)
            .attr("fill", "none")
            .attr("stroke", "grey")
            .attr("stroke-width", 1);
        group_banner.append("text")
            .attr("x", thumbnail_size * 3 / 2 + offset + 3.5)
            .attr("y", 9)
            .attr("fill", "grey")
            .style("text-anchor", "middle")
            .style("dominant-baseline", "middle")
            .text("test");
        group_banner.selectAll("text").style("cursor", "default");
    };

    that.init = function(){
        that._init();
    }.call();

    that.draw = function(){
        that.create();
        that.update();
        that.remove();
    };

    that.create = function(){
        var create_thumbnail = function(glyph, d, i) {
            var thumbnail = d3.select(glyph);
            var keypoints = d.boundary_data;
            thumbnail.attr("transform", "translate(" + 7 + ", " + (i * (thumbnail_size + offset) + 7) + ")");
            thumbnail.append("rect")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", thumbnail_size)
                .attr("height", thumbnail_size)
                .style("fill", "white")
                .style("stroke", "grey")
                .style("stroke-width", 1);
            thumbnail.append("polyline")
                .style("stroke-width", 3)
                .style("stroke", "red")
                .style("fill", "none")
                .style("stroke-linecap", "round")
                .style("storke-linejoin", "round")
                .style("pointer-events", "none")
                .style("opacity", "0.5")
                .attr("points", keypoints.reduce(
                    (total, cur) =>
                        total + String(cur[0] * thumbnail_size) + "," + String(cur[1] * thumbnail_size) + " ",
                ""));
            var btns = thumbnail.append("g")
                .attr("class", "tool-buttons")
                .style("visibility", "hidden");
            var remove_icon = btns.append("g")
                .attr("class", "snapshot-remove");
            remove_icon.append("circle")
                .attr("cx", 0)
                .attr("cy", 0)
                .attr("r", 7)
                .style("fill", "grey");
            remove_icon.append("text")
                .text("×")
                .attr("stroke", "grey")
                .attr("x", 0)
                .attr("y", 1)
                .style("text-anchor", "middle")
                .style("dominant-baseline", "middle");
            var expand_icon = btns.append("g")
                .attr("class", "snapshot-expand");
            expand_icon.append("circle")
                .attr("cx", 15)
                .attr("cy", 0)
                .attr("r", 7)
                .style("fill", "grey");
            expand_icon.append("text")
                .text("＋")
                .attr("stroke", "white")
                .attr("x", 15)
                .attr("y", 0.38)
                .style("text-anchor", "middle")
                .style("dominant-baseline", "middle");
            var compare_icon = btns.append("g")
                .attr("class", "snapshot-compare");
            compare_icon.append("circle")
                .attr("cx", 30)
                .attr("cy", 0)
                .attr("r", 7)
                .style("fill", "grey");
            compare_icon.append("text")
                .text("⤢")
                .attr("stroke", "white")
                .attr("x", 30)
                .attr("y", 0.5)
                .style("text-anchor", "middle")
                .style("dominant-baseline", "middle");
        };

        var binding_button = function(glyph, d) {
            var btns = d3.select(glyph)
                .select(".tool-buttons");
            var remove_icon = btns.select(".snapshot-remove");
            var expand_icon = btns.select(".snapshot-expand");
            var compare_icon = btns.select(".snapshot-compare");
            remove_icon.on("click", () => {
                that.remove_snapshot(d);
            });
            expand_icon.on("click", () => {
                console.log(d);
                LensView.load_snapshot(d);
            });
            compare_icon.on("click", () => {
                compare_snapshot(d);
            });
        };

        var train_snapshots = snapshots.filter(d => d.datatype === "train");
        var train_ss = train_group.selectAll("g.snapshot").data(train_snapshots);
        train_ss.enter()
            .append("g")
            .attr("class", "snapshot")
            .each(function(d, i) {
                create_thumbnail(this, d, i);
            });
        train_ss.exit().remove();
        train_group.selectAll("g.snapshot")
            .data(train_snapshots)
            .each(function(d) {
                binding_button(this, d);
            });

        var test_snapshots = snapshots.filter(d => d.datatype === "test");
        var test_ss = test_group.selectAll("g.snapshot").data(test_snapshots);
        test_ss.enter()
            .append("g")
            .attr("class", "snapshot")
            .each(function(d, i) {
                create_thumbnail(this, d, i);
            });
        test_ss.exit().remove();
        test_group.selectAll("g.snapshot")
            .data(train_snapshots)
            .each(function(d) {
                binding_button(this, d);
            });

        group_container.selectAll("g.snapshot")
            .on("mouseover", function() {
                d3.select(this)
                    .select("g.tool-buttons")
                    .style("visibility", "visible");
            })
            .on("mouseout", function() {
                d3.select(this)
                    .select("g.tool-buttons")
                    .style("visibility", "hidden");
            })
    };

    that.update = function(){
        var dot_highlight = function(glyph, d) {
            var thumbnail = d3.select(glyph).selectAll("rect.highlighted").data(d.focus);
            thumbnail.enter()
                .append("rect")
                .attr("class", "highlighted")
                .attr("x", focus => thumbnail_size * (d.sampling_area.center_x - d.sampling_area.half_sampling_len)
                    + thumbnail_size * focus.x / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .attr("y", focus => thumbnail_size * (d.sampling_area.center_y - d.sampling_area.half_sampling_len)
                    + thumbnail_size * focus.y / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .attr("width", focus => thumbnail_size * focus.size / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .attr("height", focus => thumbnail_size * focus.size / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .style("fill", "green");
            thumbnail.exit().remove();
            thumbnail.attr("x", focus => thumbnail_size * (d.sampling_area.center_x - d.sampling_area.half_sampling_len)
                    + thumbnail_size * focus.x / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .attr("y", focus => thumbnail_size * (d.sampling_area.center_y - d.sampling_area.half_sampling_len)
                    + thumbnail_size * focus.y / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .attr("width", focus => thumbnail_size * focus.size / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .attr("height", focus => thumbnail_size * focus.size / d.layout_size * 2 * d.sampling_area.half_sampling_len)
                .style("fill", "green");
        };

        train_group.selectAll("g.snapshot")
            .each(function(d) {
                dot_highlight(this, d);
            });
        test_group.selectAll("g.snapshot")
            .each(function(d) {
                dot_highlight(this, d);
            });
    };

    that.remove = function(){
    };

    that.load_data = function(data){
        snapshots.push(data);
    };

    that.remove_snapshot = function(data) {
        var index = snapshots.indexOf(data);
        snapshots.splice(index, 1);
        that.draw();
    };

    that.zoomed = function(){
        const {x, y, k} = d3.event.transform;
        let t = d3.zoomIdentity;
        t = t.translate(x, y).scale(k);
        group_container.attr("transform",t);
    };

    that.setVisible = function(visible) {
        group_container.style("visibility", visible ? "visible" : "hidden");
        if (!visible) {
            svg.attr("width", 20);
        } else {
            svg.attr("width", layout_width)
        }
    }
};