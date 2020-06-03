var MultiFocus = function(container, parent) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var plot_width = layout_width;
    var plot_height = layout_height - 20;
    plot_width = plot_height;

    // var train_focus = Array();
    // var test_focus = Array();
    // var all_focus = Array();
    var focus_data = Array();

    var group_container = that.container.append("g").attr("id", "multifocus-group");

    that._init = function(){
        group_container.attr("width", layout_width)
            .attr("height", layout_height);
    };

    that.init = function(){
        that._init();
    }.call();

    that.draw = function(){
        that.create();
        that.update();
        that.remove();
    };

    that.toggle = function(){
        focus_groups.forEach(d => {
            if (d.style("visibility") === "visible") {
                d.style("visibility", "hidden");
            } else {
                d.style("visibility", "visible");
            }
        })
    };

    that.get_current_focus = function(datatype) {
        return JSON.parse(JSON.stringify(focus_data));
    };

    that.create = function() {
        var focus_groups = group_container.selectAll("g.focus").data(focus_data);
        focus_groups.enter()
            .append("g")
            .attr("id", d => "focus_group_" + d.id)
            .attr("class", "focus")
            .attr("transform", "translate(10, 10)")
            .style("opacity", 1)
            .each(function() {
                d3.select(this)
                    .append("defs")
                    .selectAll("pattern")
                    .data(Array(9))
                    .enter()
                    .append("pattern")
                    .append("image");
                d3.select(this)
                    .selectAll("rect.multifocus")
                    .data(Array(9))
                    .enter()
                    .append("rect")
                    .attr("class", "multifocus");
                d3.select(this)
                    .append("rect")
                    .attr("x", d => d.x)
                    .attr("y", d => d.y)
                    .attr("width", d => d.size)
                    .attr("height", d => d.size)
                    .style("fill", "none")
                    .style("stroke", "green")
                    .style("stroke-width", 2);
            });
        focus_groups
            .select("rect")
            .attr("x", d => d.x)
            .attr("y", d => d.y)
            .attr("width", d => d.size)
            .attr("height", d => d.size)
            .style("fill", "none")
            .style("stroke", "green")
            .style("stroke-width", 2);
        focus_groups.exit()
            .remove();
    };

    that.update = function(){
        const pos = [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, -1],
            [-1, 1],
            [1, -1]
        ];
        group_container.selectAll("g.focus")
            .each(function(d) {
                var focus_group = d3.select(this);
                focus_group.selectAll("pattern")
                    .data(d.instances)
                    .attr("id", img => "focus-img-" + img)
                    .attr("patternUnits", "userSpaceOnUse")
                    .attr("x", d.x)
                    .attr("y", d.y)
                    .attr("width", d.size)
                    .attr("height", d.size)
                    .select("image")
                    .attr("xlink:href", img => ImageApi + "?dataset=" + Loader.dataset + "&filename=" + img + ".jpg" )
                    .attr("width", d.size)
                    .attr("height", d.size);
                focus_group.selectAll("rect.multifocus")
                    .data(d.instances)
                    .attr("x", d.x)
                    .attr("y", d.y)
                    .attr("width", d.size)
                    .attr("height", d.size)
                    .attr("rx", img => d.size / 2 * (Loader.TestData.get_cell_by_id(img)?1:0))
                    .attr("ry", img => d.size / 2 * (Loader.TestData.get_cell_by_id(img)?1:0))
                    // .attr("xlink:href", img => ImageApi + "?dataset=" + Loader.dataset + "&filename=" + img + ".jpg" )
                    .attr("fill", img => "url(#focus-img-" + img + ")")
                    .style("opacity", img => img === d.id ? 1 : 0)
                    .on("click", function() {
                        remove_multifocus_lens(parent, d);
                    });
                focus_group.on("mouseenter", function() {
                    focus_group.selectAll("rect.multifocus")
                        .transition()
                        .duration(500)
                        .attr("x", (img, i) => d.x + d.size * pos[i][0])
                        .attr("y", (img, i) => d.y + d.size * pos[i][1])
                        .style("opacity", 1);
                }).on("mouseleave", function() {
                    focus_group.selectAll("rect.multifocus")
                        .transition()
                        .duration(500)
                        .attr("x", d.x)
                        .attr("y", d.y)
                        .style("opacity", img => img === d.id ? 1 : 0);
                })
            });
    };

    that.remove = function(){
    };

    that.add_data = function(data){
        // var grid_N = parent.get_data_type() === "train" ? Loader.TrainData.get_grid_size() : Loader.TestData.get_grid_size();
        // var focus_data = parent.get_data_type() === "train" ? train_focus : test_focus;
        // var full_data = data.map((d, i) => {
        //     var grid = $("rect.boundingbox#ID-" + d);
        //     return i > 0 ? {
        //         id: d,
        //         x: Math.round(Number(grid.attr("x")) / plot_width * grid_N),
        //         y: Math.round(Number(grid.attr("y")) / plot_height * grid_N),
        //         size: 3
        //     } : {
        //         id: d,
        //         x: Math.round(Number(grid.attr("x")) / plot_width * grid_N),
        //         y: Math.round(Number(grid.attr("y")) / plot_height * grid_N),
        //         size: 5
        //     }
        // });
        //
        // var fixed = Array(data.length).fill(false);
        // var dist = Array(data.length).fill(Infinity);
        // var count = 0, cur = 0, next = -1;
        // while (count < 10) {
        //     let path = [{x: full_data[cur].x, y: full_data[cur].y}];
        //     let offset = 0, ptr = 0;
        //     while (true) {
        //         var flag = true;
        //         var pos = path[ptr];
        //         if (pos.x < 0 || pos.y < 0 || pos.x >= grid_N || pos.y > grid_N) {
        //             flag = false;
        //         } else {
        //             fixed.forEach((d, i) => {
        //                 if (cur == i || !d) {
        //                     return;
        //                 }
        //                 let separate_dist = (full_data[i].size + full_data[cur].size) / 2;
        //                 if (Math.abs(pos.x - full_data[i].x) < separate_dist &&
        //                     Math.abs(pos.y - full_data[i].y) < separate_dist) {
        //                     flag = false;
        //                 }
        //             });
        //         }
        //         if (flag) {
        //            full_data[cur].x = pos.x;
        //            full_data[cur].y = pos.y;
        //            break;
        //         } else {
        //             ++ptr;
        //             if (ptr >= path.length) {
        //                 ++offset;
        //                 for (let i = 1; i <= offset - 1; ++i) {
        //                     path.push({x: full_data[cur].x - i, y: full_data[cur].y - (offset - i)});
        //                     path.push({x: full_data[cur].x + i, y: full_data[cur].y - (offset - i)});
        //                     path.push({x: full_data[cur].x - i, y: full_data[cur].y + (offset - i)});
        //                     path.push({x: full_data[cur].x + i, y: full_data[cur].y + (offset - i)});
        //                 }
        //                 path.push({x: full_data[cur].x , y: full_data[cur].y + offset});
        //                 path.push({x: full_data[cur].x, y: full_data[cur].y - offset});
        //                 path.push({x: full_data[cur].x - offset, y: full_data[cur].y});
        //                 path.push({x: full_data[cur].x + offset, y: full_data[cur].y});
        //             }
        //         }
        //     }
        //     fixed[cur] = true;
        //     let min_dist = Infinity;
        //     full_data.forEach((d, i) => {
        //         if (!fixed[i]) {
        //             if (Math.abs(d.x - full_data[cur].x) + Math.abs(d.y - full_data[cur].y) < dist[cur]) {
        //                 dist[i] = Math.abs(d.x - full_data[cur].x) + Math.abs(d.y - full_data[cur].y);
        //             }
        //             if (dist[i] < min_dist) {
        //                 min_dist = dist[i];
        //                 next = i;
        //             }
        //         }
        //     });
        //     cur = next;
        //     ++count;
        // }
        var grid = container.select("rect.boundingbox#ID-" + data.info.id);
        var x = Number(grid.attr("x")), y = Number(grid.attr("y")), size = Number(grid.attr("width")), id = data.info.id;
        var full_data = {
            id: id,
            instances: data.similar_instances,
            x: x,
            y: y,
            center_x: x + size / 2,
            center_y: y + size / 2,
            size: size
        };
        focus_data.push(full_data)
    };

    that.remove_data = function(data) {
        var index = focus_data.indexOf(data);
        focus_data.splice(index, 1);
        that.draw();
    };

    that.load_data = function(data) {
        focus_data = Array();
        focus_data.push.apply(focus_data, data);
    };

    that.clear_data = function() {
        focus_data = Array();
    };

    that.zoomed = function(x, y, k){
        let t = d3.zoomIdentity;
        t = t.translate(x, y).scale(k);
        group_container.attr("transform",t);
    };

};
