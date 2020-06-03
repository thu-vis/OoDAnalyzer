var Recommendation = function(container, parent) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var plot_width = layout_width;
    var plot_height = layout_height - 20;
    plot_width = plot_height;
    var scaling = 1;
    var transformation = {
        x: 0,
        y: 0
    };
    var boundingbox_width = 5;
    var img_width = 100;

    var all_data = null;

    // var display_mode = "recommended";

    // var full_data = Array();
    // var recommendations = Array();

    var group_container = that.container.append("g").attr("id", "recommendation-group");
    // var tag_group = group_container.append("g").attr("id", "tag-group");
    // var recommended_group = group_container.append("g").attr("id", "recommended-group");
    // var fullimgs_group = group_container.append("g").attr("id", "fullimgs_group");
    var pattern_group = group_container.append("defs").attr("id", "pattern-group");
    var img_group = group_container.append("g").attr("id", "img-group");

    that._init = function(){
        console.log("recommendation init");
        group_container.attr("width", layout_width)
            .attr("height", layout_height);
        // recommended_group.attr("transform",
        //     "translate(" + ( 10 ) + "," + ( 10 )+ ")");
        // tag_group.attr("transform",
        //     "translate(" + ( 10 ) + "," + ( 10 )+ ")");
        // fullimgs_group.attr("transform",
        //     "translate(" + ( 10 ) + "," + ( 10 )+ ")");
        img_group.attr("transform",
            "translate(" + ( 10 ) + "," + ( 10 )+ ")");
    };

    that.init = function(){
        that._init();
    }.call();

    that.draw = function(){
        // console.log("changjian test");
        that.create();
        that.update();
        that.remove();
        // img_group.transition()
        //     .duration(1000)
        //     .attr("transform", "translate(" + ( 10 ) + "," + ( 10 ) + ")");
        // if (display_mode === "to_full") {
        //     display_mode = "full";
        // }
    };

    that.create = function(){
        // if (display_mode === "recommended") {
        //     fullimgs_group.selectAll("*").remove();
        //     var imgs = recommended_group.selectAll("image.grid").data(recommendations);
        //     imgs.enter()
        //         .append("image")
        //         .attr("class", "grid");
        //     imgs.exit().remove();
        //     var tags = tag_group.selectAll("rect.tag").data(recommendations);
        //     tags.enter()
        //         .append("rect")
        //         .attr("class", "tag");
        //     tags.exit().remove();
        // } else if (display_mode === "to_full") {
        //     recommended_group.selectAll("*").remove();
        //     tag_group.selectAll("*").remove();
        //     fullimgs_group.selectAll("image.grid")
        //         .data(full_data)
        //         .enter()
        //         .append("image")
        //         .attr("class", "grid");
        // }
        // img_group.selectAll("image.display")
        //     .data(all_data)
        //     .enter()
        //     .append("image")
        //     .attr("class", "display");
        // pattern_group.selectAll("pattern")
        //     .data(all_data)
        //     .enter()
        //     .append("pattern")
        //     .append("image");
        // img_group.selectAll("rect.display")
        //     .data(all_data)
        //     .enter()
        //     .append("rect")
        //     .attr("class", "display");
    };

    that.update = function(){
        // var grid_N = parent.get_data_type() === "train" ? Loader.TrainData.get_grid_size() : Loader.TestData.get_grid_size();
        // var grid_size = plot_width / grid_N;
        // if (display_mode === "recommended") {
        //     recommended_group.selectAll("image.grid")
        //         .attr("x", d => d.label.x)
        //         .attr("y", d => d.label.y)
        //         .attr("width", d => d.label.w)
        //         .style("opacity", 0.75)
        //         .on("mousemove", function () {
        //             d3.select(this).style("opacity", 1);
        //         })
        //         .on("mouseout", function () {
        //             d3.select(this).style("opacity", 0.75);
        //         })
        //         // .attr("xlink:href", d => d.img_url);
        //     tag_group.selectAll("rect.tag")
        //         .attr("x", d => d.grid.x)
        //         .attr("y", d => d.grid.y)
        //         .attr("width", grid_size)
        //         .attr("height", grid_size)
        //         .style("fill", "none")
        //         .style("stroke-width", 2)
        //         .style("stroke", "grey")
        //         .style("opacity", 0.5);
        // } else if (display_mode === "to_full") {
        //     var bounding = boundingbox_width / scaling;
        //     fullimgs_group.selectAll("image.grid")
        //         .attr("x", d => d.x * grid_size + bounding)
        //         .attr("y", d => d.y * grid_size + bounding)
        //         .attr("width", grid_size - 2 * bounding)
        //         .style("opacity", 0.75)
        //         .on("mousemove", function () {
        //             d3.select(this).style("opacity", 1);
        //         })
        //         .on("mouseout", function () {
        //             d3.select(this).style("opacity", 0.75);
        //         })
        //         // .attr("xlink:href", d => d.img_url);
        // }
        // img_group.selectAll("image.display")
        //     .data(all_data)
        //     .attr("x", d => d.grid_x[0] * plot_width + boundingbox_width)
        //     .attr("y", d => d.grid_x[1] * plot_height + boundingbox_width)
        //     .attr("width", d => d.width * plot_width - 2 * boundingbox_width)
        //     .attr("height", d => d.width * plot_height - 2 * boundingbox_width)
        //     .attr("xlink:href", d => ImageApi + "?dataset=" + Loader.dataset + "&filename=" + d.get_id() + ".jpg")
        //     // .style("opacity", 0.5)
        //     .style("pointer-events", "none");

        // var patterns = pattern_group.selectAll("pattern").data(all_data, d => d.get_id());
        // var images = img_group.selectAll("rect.display").data(all_data, d => d.get_id());
        //
        // pattern_group.selectAll("pattern")
        //     .data(all_data)
        //     .attr("id", d => "radius-img-" + d.get_id())
        //     .attr("patternUnits", "userSpaceOnUse")
        //     .attr("x", d => d.grid_x[0] * plot_width + 0.5 * boundingbox_width)
        //     .attr("y", d => d.grid_x[1] * plot_width + 0.5 * boundingbox_width)
        //     .attr("width", d => d.width * plot_width - boundingbox_width)
        //     .attr("height", d => d.width * plot_height - boundingbox_width)
        //     .select("image")
        //     .attr("xlink:href", function(d){
        //         let width = d.width * plot_width - boundingbox_width;
        //         if (width < 50){
        //             return d.get_thumbnail_url();
        //         }
        //         else{
        //             return d.get_img_url();
        //         }
        //     })
        //     .attr("width", d => d.width * plot_width - boundingbox_width)
        //     .attr("height", d => d.width * plot_height - boundingbox_width);
        // img_group.selectAll("rect.display")
        //     .data(all_data)
        //     .attr("x", d => d.grid_x[0] * plot_width + 0.5 * boundingbox_width)
        //     .attr("y", d => d.grid_x[1] * plot_height + 0.5 * boundingbox_width)
        //     .attr("width", d => d.width * plot_width - boundingbox_width)
        //     .attr("height", d => d.width * plot_height - boundingbox_width)
        //     .attr("rx", d => (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "test"))
        //     .attr("ry", d => (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "test"))
        //     .attr("fill", d => "url(#radius-img-" + d.get_id() + ")")
        //     // .style("opacity", 0.5)
        //     .style("pointer-events", "none");
    };

    that.remove = function(){
        // pattern_group.selectAll("pattern").data(all_data).exit().remove();
        // img_group.selectAll("rect.display").data(all_data).exit().remove();
    };

    that.layout = function() {
        var grid_N = parent.get_data_type() === "train" ? Loader.TrainData.get_grid_size() : Loader.TestData.get_grid_size();
        var grid_size = plot_width / grid_N;
        var img_size = img_width / scaling;
        // recommendations = Array();
        // var importance_threshold = 0.5;
        // for (let d of full_data) {
        //     if (d.entropy < importance_threshold) {
        //         break;
        //     }
        //     // var layout_size = d.entropy > 0.6 ? 7 : 5;
        //     var layout_size = 5;
        //     var flag = recommendations.reduce((res, cur) =>
        //         res && (Math.abs(cur.x - d.x) >= (layout_size + cur.size) / 2 || Math.abs(cur.y - d.y) >= (layout_size + cur.size) / 2), true);
        //     if (flag) {
        //         recommendations.push({
        //             ...d,
        //             size: layout_size
        //         });
        //     }
        // }
        var intersect = function(rect1, rect2) {
            return !((rect1.x + rect1.w < rect2.x) || (rect2.x + rect2.w < rect1.x) ||
                (rect1.y + rect1.h < rect2.y) || (rect2.y + rect2.h < rect1.y))
        };
        var legal = function(rect) {
            return rect.x > 0 &&
                rect.y > 0 &&
                rect.x + rect.w < plot_width &&
                rect.y + rect.h < plot_height;
        };
        const offset = [
            { x: 0, y: 0 },
            { x: 0, y: -img_size },
            { x: -img_size, y: -img_size },
            { x: -img_size, y: 0 }
        ];
        for (let d of full_data) {
            var center = {
                x: (d.x + 0.5) * grid_size,
                y: (d.y + 0.5) * grid_size,
            };
            var tmp_grid = {
                x: d.x * grid_size,
                y: d.y * grid_size,
                w: grid_size,
                h: grid_size
            };
            var tmp_label;
            var can_placed;
            if (d.label != null) {
                tmp_label = {
                    dir: d.label.dir,
                    x: center.x + offset[d.label.dir].x,
                    y: center.y + offset[d.label.dir].y,
                    w: img_size,
                    h: img_size
                };
                can_placed = true;
                for (let r of recommendations) {
                    if (intersect(tmp_label ,r.label) || intersect(tmp_grid ,r.label) ||
                        intersect(tmp_label ,r.grid) || intersect(tmp_grid ,r.grid) ||
                        !legal(tmp_label)) {
                        can_placed = false;
                    }
                }
                if (can_placed) {
                    d.label = tmp_label;
                    recommendations.push({
                        img_url: d.img_url,
                        label: tmp_label,
                        grid: tmp_grid
                    });
                } else {
                    d.label = null;
                }
            } else {
                for (var i = 0; i < 4; ++i) {
                    tmp_label = {
                        dir: i,
                        x: center.x + offset[i].x,
                        y: center.y + offset[i].y,
                        w: img_size,
                        h: img_size
                    };
                    can_placed = true;
                    for (let r of recommendations) {
                        if (intersect(tmp_label, r.label) || intersect(tmp_grid, r.label) ||
                            intersect(tmp_label, r.grid) || intersect(tmp_grid, r.grid) ||
                            !legal(tmp_label)) {
                            can_placed = false;
                            break;
                        }
                    }
                    if (can_placed) {
                        d.label = tmp_label;
                        recommendations.push({
                            img_url: d.img_url,
                            label: d.label,
                            grid: tmp_grid
                        });
                        break;
                    }
                    d.label = null;
                }
            }
        }
        console.log(recommendations);
    };

    that.load_data = function(data){
        // var importance_compare = (data_1, data_2) =>
        //     data_1.entropy < data_2.entropy ? 1 : data_1.entropy === data_2.entropy ? 0 : -1;
        // full_data = data.get_all_data().map((d, i) => Object({
        //     idx: i,
        //     x: d.get_coord()[0],
        //     y: d.get_coord()[1],
        //     entropy: d.get_entropy(),
        //     img_url: data.get_cell(i).get_img_url(),
        //     label: null
        // }));
        // full_data.sort(importance_compare);
        // if (scaling <= 2.5) {
        //     display_mode = "recommended";
        //     that.layout();
        // } else {
        //     display_mode = "to_full";
        // }
        // console.log(data);
        all_data = parent.get_data();
    };

    that.zoomed = function(x, y, k){
        let t = d3.zoomIdentity;
        t = t.translate(x, y).scale(k);
        transformation = {
            x: x,
            y: y
        };
        scaling = k;
        group_container.attr("transform",t);
        // if (scaling <= 1) {
        //     display_mode = "recommended";
        //     that.layout();
        // } else {
        //     display_mode = "to_full";
        // }
        that.draw();
    };

    that.setVisible = function(visible) {
        group_container.style("visibility", visible ? "visible" : "hidden");
    };
};