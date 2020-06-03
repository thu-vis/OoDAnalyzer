/**
 * Created by Changjian on 2018/12/17.
 */

var SingleGridLayout = function(container, labelimage_container, data_type){
    let that = this;
    that.container = container;
    that.labelimage_container = labelimage_container;
    that.data_type = data_type;

    let boundingbox_width = 0.05;

    let group_container = that.container;
    let plot_group = group_container.append("g").attr("id", "plot-group");
    let pattern_group = group_container.append("defs").attr("id", "pattern-group");

    let plot_width = null;
    let plot_height = null;
    let all_data = null;
    let labels = Array();
    let rects = null;

    let label_layout_mode = 1;
    let img_visible = "visible";

    that._init = function(){
        // plot_group.attr("transform",
        //     "translate(" + ( margin_size ) + "," + ( margin_size )+ ")");
        // that.labelimage_container.attr("transform",
        //     "translate(" + ( margin_size ) + "," + ( margin_size )+ ")");
    }.call();

    that._create = function(){
        that.grid_group_create();
    };
    that._update = function(){
        let flag_exit_zero = that.grid_group_update();
        return flag_exit_zero;
    };
    that._remove = function(){
        that.grid_group_remove();
    };

    that.grid_group_create = function(){

        pattern_group.selectAll("pattern")
            .data(all_data)
            .enter()
            .append("pattern")
            .append("image");

        rects.enter()
            .append("rect")
            .attr("id", d => "ID-" + d.get_id())
            .attr("class", "boundingbox")
            .attr("x", d => d.grid_x[0] * plot_width)
            .attr("y", d => d.grid_x[1] * plot_width)
            .attr("width", d => d.width * plot_width)
            .attr("height", d => d.width * plot_width)
            .attr("rx", d => d.width * plot_width / 2 * (d.get_datatype() === "train"))
            .attr("ry", d => d.width * plot_width / 2 * (d.get_datatype() === "train"))
            .style("fill", d => CategoryColor[d.get_y()])
            .style("stroke", "black")
            .style("stroke-width", 0)
            .style("stroke-opacity", 1)
            .style("opacity", 0)
            .transition()
            .duration(AnimationDuration)
            .delay(AnimationDuration * 2)
            .attr("x", d => d.grid_x[0] * plot_width)
            .attr("y", d => d.grid_x[1] * plot_width)
            .attr("width", d => d.width * plot_width)
            .attr("height", d => d.width * plot_width)
            .attr("rx", d => d.width * plot_width / 2 * (d.get_datatype() === "train"))
            .attr("ry", d => d.width * plot_width / 2 * (d.get_datatype() === "train"))
            .style("fill", d => CategoryColor[d.get_y()])
            .style("opacity", 1);

    };
    that.grid_group_update = function(){
        // rects.transition()
        //     .duration(AnimationDuration)
        //     .delay(AnimationDuration)
        //     .attr("x", d => d.grid_x[0] * plot_width)
        //     .attr("y", d => d.grid_x[1] * plot_width)
        //     .attr("width", d => d.width * plot_width)
        //     .attr("height", d => d.width * plot_width)
        //     .attr("rx", d => d.width * plot_width / 2 * (d.get_datatype() === "train"))
        //     .attr("ry", d => d.width * plot_width / 2 * (d.get_datatype() === "train"))
        //     .style("fill", d => CategoryColor[d.get_y()])
        //     .style("opacity", 1);
        //
        //
        // pattern_group.selectAll("pattern")
        //     .data(all_data)
        //     .attr("id", d => "radius-img-" + d.get_id())
        //     .attr("patternUnits", "userSpaceOnUse")
        //     // .attr("x", d => d.grid_x[0] * plot_width + 0.5 * boundingbox_width)
        //     // .attr("y", d => d.grid_x[1] * plot_width + 0.5 * boundingbox_width)
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

        console.log("distribution update");
        boundingbox_width = 0.01 * plot_width;
        let flag_exit_zero = false;

        var grids = plot_group.selectAll("g.grid").data(all_data, d => d.get_id());
        var enter_size = grids.enter().size(),
            update_size = grids.size(),
            exit_size = grids.exit().size();
        var cumulated_animation_time = 0,
            single_animation_time = 500;

        var patterns = pattern_group.selectAll("pattern").data(all_data, d => d.get_id());
        var imglabels = that.labelimage_container.selectAll("g").data(labels, d => d.get_id());
        console.log(labels);
        // Animation
        if (exit_size > 0) {
            flag_exit_zero = true;
            patterns.exit()
                .transition()
                .duration(single_animation_time)
                .remove();
            grids.exit()
                .transition()
                .duration(single_animation_time)
                .style("opacity", 0)
                .remove();

            cumulated_animation_time += single_animation_time * 1.1;
            console.log("in exit size");
        }
        imglabels.exit()
            .transition()
            .duration(single_animation_time)
            .style("opacity", 0)
            .remove();
        imglabels.style("visibility", img_visible);
        imglabels.select("rect")
            .transition()
            .duration(single_animation_time * 4)
            .delay(cumulated_animation_time)
            .attr("x", d => d.grid.x)
            .attr("y", d => d.grid.y)
            .attr("width", d => d.grid.w)
            .attr("height", d => d.grid.h)
            .attr("rx", d => d.grid.w / 2 * (d.get_datatype() === "train"))
            .attr("ry", d => d.grid.h / 2 * (d.get_datatype() === "train"));
        imglabels.select("image")
            .transition()
            .duration(single_animation_time * 4)
            .delay(cumulated_animation_time)
            .attr("x", d => d.label.x)
            .attr("y", d => d.label.y)
            .attr("width", d => d.label.w)
            .attr("height", d => d.label.h)
            .attr("xlink:href", d => d.get_img_url());
        if (update_size > 0) {
            patterns.attr("id", d => "radius-img-" + d.get_id())
                .transition()
                .duration(single_animation_time)
                .delay(cumulated_animation_time)
                .attr("x", 0.5 * boundingbox_width)
                .attr("y", 0.5 * boundingbox_width)
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width)
                .select("image")
                .attr("xlink:href", function(d){
                    if (label_layout_mode === 0) {
                        let width = d.width * plot_width - boundingbox_width;
                        if (width < 30){
                            return d.get_thumbnail_url();
                        }
                        else{
                            return d.get_img_url();
                        }
                    }
                    else {
                        return null;
                    }
                })
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width);
            grids.attr("id", d => "ID-" + d.get_id())
                .each(function(d) {
                   d3.select(this)
                        .select(".boundingbox")
                        .transition()
                        .duration(single_animation_time * 4)
                        .delay(cumulated_animation_time)
                        .attr("width", d.width * plot_width)
                        .attr("height", d.width * plot_width)
                        .attr("rx", d.width * plot_width / 2 * (d.get_datatype() === "train"))
                        .attr("ry", d.width * plot_width / 2 * (d.get_datatype() === "train"));
                   d3.select(this)
                        .select(".display")
                        .transition()
                        .duration(single_animation_time * 4)
                        .delay(cumulated_animation_time)
                        .attr("width", d.width * plot_width - boundingbox_width)
                        .attr("height", d.width * plot_height - boundingbox_width)
                        .attr("rx", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .attr("ry", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .style("visibility", img_visible)
                        .attr("fill", "url(#radius-img-" + d.get_id() + ")");
                })
                .transition()
                .duration(single_animation_time * 4)
                .delay(cumulated_animation_time)
                .attr("transform", d => "translate(" + (d.grid_x[0] * plot_width) + ", " +  (d.grid_x[1] * plot_width) + ")")
                .attr("width", d => d.width * plot_width)
                .attr("height", d => d.width * plot_width);

            cumulated_animation_time += single_animation_time * 4.1;
        }
        if (enter_size > 0) {
            patterns.enter()
                .append("pattern")
                .attr("id", d => "radius-img-" + d.get_id())
                .attr("patternUnits", "userSpaceOnUse")
                .attr("x", 0.5 * boundingbox_width)
                .attr("y", 0.5 * boundingbox_width)
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width)
                .append("image")
                .attr("xlink:href", function(d){
                    if (label_layout_mode === 0) {
                        let width = d.width * plot_width - boundingbox_width;
                        if (width < 30){
                            return d.get_thumbnail_url();
                        }
                        else{
                            return d.get_img_url();
                        }
                    }
                    else {
                        return null;
                    }
                })
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width);
            grids.enter()
                .append("g")
                .attr("id", d => "ID-" + d.get_id())
                .attr("class", "grid")
                .attr("transform", d => "translate(" + (d.grid_x[0] * plot_width) + ", " +  (d.grid_x[1] * plot_width) + ")")
                .attr("width", d => d.width * plot_width)
                .attr("height", d => d.width * plot_width)
                .style("opacity", 0)
                .on('click', function(d) {
                    InfoView.load_data([d.get_id()]);
                    InfoView.draw_images();
                })
                .on("mousemove", function() {
                    d3.select(this).select("rect")
                        .style("stroke-width",  2.0);
                })
                .on("mouseout", function() {
                    d3.select(this).select("rect")
                        .style("stroke-width", 0.0);
                })
                .on("mouseenter", function(d){
                    d3.select(this).select("rect")
                        .style("stroke-width",  2.0);
                    // tips_in_another(d.get_feature(), true, that.data_type);
                })
                .on("mouseleave", function(d){
                    d3.select(this).select("rect")
                        .style("stroke-width",  0.0);
                    // tips_in_another(d.get_feature(), false, that.data_type);
                })
                .each(function(d) {
                    d3.select(this)
                        .append("rect")
                        .attr("class", "boundingbox")
                        .attr("x", 0)
                        .attr("y", 0)
                        .attr("width", d.width * plot_width)
                        .attr("height", d.width * plot_width)
                        .attr("rx", d.width * plot_width / 2 * (d.get_datatype() === "train"))
                        .attr("ry", d.width * plot_width / 2 * (d.get_datatype() === "train"))
                        .style("fill", CategoryColor[d.get_y()])
                        .style("stroke", "black")
                        .style("stroke-width", 0)
                        .style("stroke-opacity", 1);
                    d3.select(this)
                        .append("rect")
                        .attr("class", "display")
                        .attr("x", 0.5 * boundingbox_width)
                        .attr("y", 0.5 * boundingbox_width)
                        .attr("width", d.width * plot_width - boundingbox_width)
                        .attr("height", d.width * plot_height - boundingbox_width)
                        .attr("rx", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .attr("ry", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .attr("fill", "url(#radius-img-" + d.get_id() + ")")
                        .style("visibility", img_visible)
                        .style("pointer-events", "none");

                })
                .transition()
                .duration(single_animation_time)
                .delay(cumulated_animation_time)
                .style("opacity", 1);
        }
        var imglabels_g = imglabels.enter().append("g").style("visibility", img_visible);
        imglabels_g.append("rect")
            .attr("x", d => d.grid.x)
            .attr("y", d => d.grid.y)
            .attr("width", d => d.grid.w)
            .attr("height", d => d.grid.h)
            .attr("rx", d => d.grid.w / 2 * (d.get_datatype() === "train"))
            .attr("ry", d => d.grid.h / 2 * (d.get_datatype() === "train"))
            .style("fill", "none")
            .style("stroke", "grey")
            .style("stroke-width", 2)
            .style("opacity", 0)
            .transition()
            .duration(single_animation_time)
            .delay(cumulated_animation_time)
            .style("opacity", 1);
        imglabels_g.append("image")
            .attr("x", d => d.label.x)
            .attr("y", d => d.label.y)
            .attr("width", d => d.label.w)
            .attr("height", d => d.label.h)
            .attr("xlink:href", d => d.get_img_url())
            .style("opacity", 0)
            .transition()
            .duration(single_animation_time)
            .delay(cumulated_animation_time)
            .style("opacity", 1);
        return flag_exit_zero;
    };
    that.grid_group_remove = function(){
        rects.exit()
            .transition()
            .duration(AnimationDuration)
            .style("opacity", 0)
            .remove();

        pattern_group.selectAll("pattern").data(all_data).exit().remove();
    };

    that.update = function(data, _labels, container_left_x, container_top_y, _plot_width = 500, margin_size = 0, margin_top_size=0, first_time_flag=false){
        // update the view according the position and id information in data
        console.log("grid layout view update");
        all_data = data;
        plot_width = _plot_width;
        plot_height =_plot_width;
        if (_labels !== null) {
            labels = _labels;
            label_layout_mode = 1
        } else {
            labels = Array();
            label_layout_mode = 0;
        }
        // this._create();
        let flag_exit_zero = this._update();
        // this._remove();
        let delay = 0;
        // if (1) {delay = AnimationDuration * 1.1;}
        let animation_duration = 0;
        if (first_time_flag){
            animation_duration = 0;
        }
        else{
            animation_duration = AnimationDuration;
        }

        group_container
            .transition()
            .duration(animation_duration * 4.1)
            .delay(animation_duration * 1.1)
            .attr("transform", "translate(" + ( container_left_x + margin_size ) + "," +
                ( container_top_y + margin_top_size )+ ")");
        that.labelimage_container
            .transition()
            .duration(animation_duration * 4.1)
            .delay(animation_duration * 1.1)
            .attr("transform", "translate(" + ( container_left_x + margin_size ) + "," +
                ( container_top_y + margin_top_size)+ ")");
    };

    that.setImagevisible = function(visible){
        img_visible = visible;
        that.labelimage_container.selectAll("g").style("visibility", img_visible);
        plot_group.selectAll("rect.display").style("visibility", img_visible);
    };

    that.get_data = function(){
        return all_data;
    };
    that.highlight_grid = function(id){
        plot_group.select("#ID-" + id).select("rect")
            .style("stroke-width", 2.0);
    };
    that.dehighlight_grid = function(id){
        plot_group.select("#ID-" + id).select("rect")
            .style("stroke-width", 0.0);
    };
};

var DistributionLayout = function(container, parent){
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var plot_width = layout_width;
    var plot_height = layout_height - 20;
    var solid_point_radius = 2;
    var text_px = 12;
    var boundingbox_width = 3;
    var first_time_flag = true;
    console.log("plot width, plot height: ", plot_width, plot_height);

    const info_panel_width = $("#info-panel")[0].offsetWidth;
    var large_grid_width = layout_height < (layout_width - info_panel_width - 20) ?
        layout_height : (layout_width - info_panel_width - 20);
    var margin_size = (layout_width - large_grid_width - info_panel_width) / 2;
    var margin_top_size = (bbox.height - large_grid_width) / 2;
    var small_grid_width = (layout_width - 3 * margin_size) / 2;
    var large_x = margin_size; // position of grid when the mode is "single" or "superposition"
    var large_y = margin_top_size;
    var small_x_1 = 0; // position of the left grid when the mode is "juxtaposition"
    var small_y_1 = 100;
    var small_x_2 = small_grid_width + margin_size; // position of the right grid when the mode is "juxtaposition"
    var small_y_2 = 100;


    let labels = Array(); // Label layout
    let img_width = 40;


    var mouse_pressed = false;
    var mouse_pos = {
        x: -1,
        y: -1
    };

    var sampling_area = {
        x: 0,
        y: 0,
        w: 1,
        h: 1
    };
    var relative_sampling_area = {
        x: 0,
        y: 0,
        w: 1,
        h: 1
    };

    var plot_x, plot_y, plot_w, plot_h, ratio = 1;

    // data
    let train_data = Array(), test_data = Array();
    let train_boundary_points = Array(), test_boundary_points = Array(), all_boundary_points = Array();
    let train_data_svg_items = Array(), test_data_svg_items = Array();
    let train_id = -1, test_id = -1, all_id = -1, data_id = -1;

    // flag
    let image_visible = "visible";
    // views

    var group_container = that.container.append("g").attr("id", "distribution-group");
    let training_group = group_container.append("g").attr("id", "training-group");
    let test_group = group_container.append("g").attr("id", "test-group");
    let labelimage_training_group = group_container.append("g").attr("id", "labelimage_training_group");
    let labelimage_test_group = group_container.append("g").attr("id", "labelimage_test_group");
    var overview_group = group_container.append("g").attr("id", "overview-group");
    let confirm_button;

    that.training_view = new SingleGridLayout(training_group, labelimage_training_group, "training");
    that.test_view = new SingleGridLayout(test_group, labelimage_test_group, "test");

    that._init = function(){
        // set up container's width and height
        d3.select("#preloader")
            .style("top", large_grid_width / 2 + margin_size)
            .style("margin-left", large_grid_width / 2 + margin_size)
			.style("display", "block");

        group_container.attr("width", layout_width)
            .attr("height", layout_height);

        overview_group.attr("transform",
            "translate(" + ( margin_size ) + "," + ( margin_top_size )+ ")")
            .style("visibility", "hidden");
        overview_group.append("rect")
            .attr("id", "overview-1")
            .attr("class", "overview-box");
        overview_group.append("rect")
            .attr("id", "overview-2")
            .attr("class", "overview-box");
        overview_group.selectAll(".overview-box")
            .attr("x", 0)
            .attr("y", 0)
            .style("fill", "white")
            .style("stroke", "grey")
            .style("stroke-width", 5)
            .style("opacity", 0.3);
        overview_group.append("rect")
            .attr("id", "viewbox")
            .style("stroke-dasharray", "5, 5")
            .style("fill", "white")
            .style("stroke", "grey")
            .style("stroke-width", 5)
            .style("opacity", 0.5);

        overview_group.on("mousedown", function() {
            var offset = $(d3.select(this).node()).offset();
            plot_x = offset.left;
            plot_y = offset.top;
            plot_w = plot_width;
            plot_h = plot_height;
            var transform  = group_container.attr("transform");
            if (transform) {
                ratio = Number(scale.slice(scale.indexOf("(") + 1, scale.indexOf(')')));
                plot_w = plot_width * ratio;
                plot_h = plot_height * ratio;
            } else {
                ratio = 1;
            }
            mouse_pos = {
                x: d3.event.pageX,
                y: d3.event.pageY
            };
            mouse_pressed = d3.select(this).attr("id");
            overview_group.select("#viewbox").style("visibility", "visible");
            confirm_button.style("visibility", "hidden");
            adjust_sampling_area(compute_viewbox(mouse_pos.x, mouse_pos.y, mouse_pos.x, mouse_pos.y));
        }).on("mousemove", function() {
            if (!mouse_pressed) {
                return;
            }
            adjust_sampling_area(compute_viewbox(mouse_pos.x, mouse_pos.y, d3.event.pageX, d3.event.pageY));

            let left_x = relative_sampling_area.x;
            let top_y = relative_sampling_area.y;
            let right_x = left_x + relative_sampling_area.w;
            let bottom_y = top_y + relative_sampling_area.h;

            if (parent.get_mode() === "selecting") {
                if (parent.get_position_mode() !== "juxtaposition") {
                    for (let i = 0; i < train_data.length; i++) {
                        let data_item = train_data[i];
                        let position = data_item.grid_x;
                        let width = data_item.width;
                        let item = train_data_svg_items[i];

                        if (position[0] + width > left_x && position[0] < right_x && position[1] + width > top_y && position[1] < bottom_y) {
                            if (data_item.selected === false) {
                                data_item.selected = true;
                                let color = d3.rgb(item.style('fill'));
                                data_item.color = color;
                                item.style('fill', color.darker(1.5));
                            }
                        }
                        else {
                            data_item.selected = false;
                            if (data_item.color !== undefined) {
                                item.style('fill', data_item.color);
                            }
                        }
                    }
                    for (let i = 0; i < test_data.length; i++) {
                        let data_item = test_data[i];
                        let position = data_item.grid_x;
                        let width = data_item.width;
                        let item = test_data_svg_items[i];

                        if (position[0] + width > left_x && position[0] < right_x && position[1] + width > top_y && position[1] < bottom_y) {
                            if (data_item.selected === false) {
                                data_item.selected = true;
                                //highlight_items_id.push(highlight_data_item_id);

                                let color = d3.rgb(item.style('fill'));
                                // console.log('color');
                                // console.log(color);
                                data_item.color = color;
                                item.style('fill', color.darker(1.5));
                            }
                        }
                        else {
                            data_item.selected = false;
                            if (data_item.color !== undefined) {
                                item.style('fill', data_item.color);
                            }
                        }
                    }
                }
                // else {
                //     if (left_x > 1){
                //         left_x -= small_x_2 / plot_width;
                //         right_x -= small_x_2 / plot_width;
                //         for (let i = 0; i < train_data.length; i++){
                //             let data_item = train_data[i];
                //             let item = train_data_svg_items[i];
                //
                //             data_item.selected = false;
                //             if(data_item.color !== undefined){
                //                 item.style('fill', data_item.color);
                //             }
                //         }
                //
                //         for (let i = 0; i < test_data.length; i++){
                //             let data_item = test_data[i];
                //             let position = data_item.grid_x;
                //             let width = data_item.width;
                //             let item = test_data_svg_items[i];
                //
                //             if (position[0] > left_x && position[0] + width < right_x && position[1] > top_y && position[1] + width < bottom_y){
                //                 if (data_item.selected === false) {
                //                     data_item.selected = true;
                //                     let color = d3.rgb(item.style('fill'));
                //                     data_item.color = color;
                //                     item.style('fill', color.darker(1.5));
                //                 }
                //             }
                //             else {
                //                 data_item.selected = false;
                //                 if(data_item.color !== undefined){
                //                     item.style('fill', data_item.color);
                //                 }
                //             }
                //         }
                //     }
                //     else {
                //         for (let i = 0; i < test_data.length; i++){
                //             let data_item = test_data[i];
                //             let item = test_data_svg_items[i];
                //
                //             data_item.selected = false;
                //             if(data_item.color !== undefined){
                //                 item.style('fill', data_item.color);
                //             }
                //         }
                //
                //         for (let i = 0; i < train_data.length; i++){
                //             let data_item = train_data[i];
                //             let position = data_item.grid_x;
                //             let width = data_item.width;
                //             let item = train_data_svg_items[i];
                //
                //             if (position[0] > left_x && position[0] + width < right_x && position[1] > top_y && position[1] + width < bottom_y) {
                //                 if (data_item.selected === false) {
                //                     data_item.selected = true;
                //                     let color = d3.rgb(item.style('fill'));
                //                     data_item.color = color;
                //                     item.style('fill', color.darker(1.5));
                //                 }
                //             }
                //             else {
                //                 data_item.selected = false;
                //                 if(data_item.color !== undefined){
                //                     item.style('fill', data_item.color);
                //                 }
                //             }
                //         }
                //     }
                // }
            }
        }).on("mouseup", function() {
            if (!mouse_pressed) {
                return;
            }
            mouse_pressed = false;
            adjust_sampling_area(compute_viewbox(mouse_pos.x, mouse_pos.y, d3.event.pageX, d3.event.pageY));
            confirm_button.attr("transform",
                "translate(" + ((relative_sampling_area.x + relative_sampling_area.w) * plot_width + margin_size) + ", " +
                ((relative_sampling_area.y + relative_sampling_area.h) * plot_width  + margin_top_size + small_y_1 * (parent.get_position_mode() === "juxtaposition")) + ")")
                .style("visibility", "visible");
        });

        confirm_button = group_container.append("g").attr("id", "confirm-resample").style("visibility", "hidden");
        confirm_button.append("circle")
            .attr("r", 20)
            .attr("fill", "grey");
        // confirm_button.append("text")
        //     .attr("text-anchor", "middle")
        //     .attr("dominant-baseline", "middle")
        //     .style("stroke", "white")
        //     .style("opacity", 1)
        //     .text("✔");
        confirm_button.append("text")
            .attr("class", 'glyphicon')
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr('dy', '0.25em')
            .style("fill", "white")
            .style("opacity", 1)
            .style('font-size', '20px')
            .text('\ue015');

        // confirm_button
        //     .append("text")
        //     .attr("transform", "translate(50, 50)")
        //     //引入FontAwesome字体
        //     .attr("class", "fa")
        //     .style("fill", "red")
        //     .text("\uf00e");
        //<use xlink:href="#favorite"></use>
        // confirm_button
        //     .append("use")
        //     .attr("xlink:href", "#favorite");

        confirm_button.on("click", function() {
            if (parent.get_mode() === "cropping") {
                sampling_area = {
                    x: sampling_area.x + sampling_area.w * relative_sampling_area.x,
                    y: sampling_area.y + sampling_area.h * relative_sampling_area.y,
                    w: sampling_area.w * relative_sampling_area.w,
                    h: sampling_area.h * relative_sampling_area.h
                };
                that.resample();
            } else if (parent.get_mode() === "selecting") {
                let selected_items_id = [];
                for (let i = 0; i < train_data.length; i++) {
                    if (train_data[i].selected === true) {
                        selected_items_id.push(train_data[i].get_id());
                    }
                }
                for (let i = 0; i < test_data.length; i++) {
                    if (test_data[i].selected === true) {
                        selected_items_id.push(test_data[i].get_id());
                    }
                }
                InfoView.load_data(selected_items_id);
                InfoView.draw_images();
            }
            set_exploring();
            d3.select(this).style("visibility", "hidden");

        });

    };

    that.init = function(){
        that._init();
    }.call();

    that.load_data = function(id, data, boundary_points){
        // TODO: each data should show its identify (belongs to training set or test set)
        let position_mode = parent.get_position_mode();
        if (position_mode === "juxtaposition") {
            if (data[0].get_datatype() === "train") {
                train_data = data;
                train_boundary_points = boundary_points;
                train_id = id;
            } else {
                test_data = data;
                test_boundary_points = boundary_points;
                test_id = id;
            }
        } else if (position_mode === "superposition"){
            train_data = Array();
            test_data = Array();
            for (let d of data) {
                if (d.get_datatype() === "train") {
                    train_data.push(d);
                } else {
                    test_data.push(d);
                }
            }
            all_boundary_points = boundary_points;
            all_id = id;
        } else {
            if (data[0].get_datatype() === "train") {
                train_data = data;
                train_boundary_points  = boundary_points;
                train_id = id;
                test_data = Array();
                test_id = -1;
            } else {
                test_data = data;
                test_boundary_points = boundary_points;
                test_id = id;
                train_data = Array();
                train_id = -1;
            }
        }
    };

    that.draw = function(){
        // called after load data

        //TODO: 1. processing data

        let training_left_x = 0;
        let training_top_y = 0;
        let test_left_x = 0;
        let test_top_y = 0;
        let training_width = large_grid_width;
        let test_width = large_grid_width;
        let position_mode = parent.get_position_mode();

        if (position_mode === "juxtaposition") {
            InfoView.close();
            training_left_x = small_x_1;
            test_left_x = small_x_2;
            training_top_y = small_y_1;
            test_top_y = small_y_2;
            training_width = small_grid_width;
            test_width = small_grid_width;
            d3.select("#preloader")
                .style("top", small_grid_width / 2 + small_y_1 + margin_top_size)
                .style("margin-left", layout_width / 2);
        } else {
            InfoView.open();
            d3.select("#preloader")
                .style("top", large_grid_width / 2 + margin_top_size)
                .style("margin-left", large_grid_width / 2 + margin_size);
        }
        plot_width = training_width;
        plot_height = plot_width;

        let train_labels, test_labels;
        if (position_mode !== "superposition") {
            train_labels = that.layout(train_data, training_width, train_boundary_points);
            test_labels = that.layout(test_data, test_width, test_boundary_points);
        } else {
            let labels = that.layout(train_data.concat(test_data), large_grid_width, all_boundary_points);
            if (labels !== null) {
                train_labels = labels.filter(d => d.get_datatype() === "train");
                test_labels = labels.filter(d => d.get_datatype() === "test");
            } else {
                train_labels = null;
                test_labels = null;
            }
        }
        if (train_labels) {
            labels.concat(train_labels);
        }
        if (test_labels) {
            labels.concat(test_labels);
        }

        // TODO:
        // Label layout result

        // Hidden overview_group
        overview_group.style("visibility", "hidden");
        container.select("#viewbox").style("visibility", "hidden");
        container.select("#confirm-resample").style("visibility", "hidden");

        //TODO: 2. call training view and test view
        // set_exploring();
        that.training_view.update(train_data, train_labels, training_left_x, training_top_y, training_width, margin_size, margin_top_size, first_time_flag);
        that.test_view.update(test_data, test_labels, test_left_x, test_top_y, test_width, margin_size, margin_top_size, first_time_flag);
        first_time_flag = false;

        let train_group = d3.select('#distribution-group #training-group #plot-group');
        let test_group = d3.select('#distribution-group #test-group #plot-group');

        train_data_svg_items = new Array();
        test_data_svg_items = new Array();

        for (let i = 0; i < train_data.length; i++) {
            train_data[i].selected = false;
            let highlight_data_item_id = train_data[i].get_id();
            let item = train_group.select('#' + 'ID-' + highlight_data_item_id + ' .boundingbox');
            train_data_svg_items.push(item);
        }
        for (let i = 0; i < test_data.length; i++) {
            test_data[i].selected = false;
            let highlight_data_item_id = test_data[i].get_id();
            let item = test_group.select('#' + 'ID-' + highlight_data_item_id + ' .boundingbox');
            test_data_svg_items.push(item);
        }
    };

    function adjust_sampling_area(area) {
        relative_sampling_area = area;
        overview_group.select("#viewbox")
            .attr("x", relative_sampling_area.x * plot_width)
            .attr("y", relative_sampling_area.y * plot_height + small_y_1 * (parent.get_position_mode() === "juxtaposition"))
            .attr("width", relative_sampling_area.w * plot_width)
            .attr("height", relative_sampling_area.h * plot_height);
    }
    function compute_viewbox(x1, y1, x2, y2) {
        var min_x = Math.min(x1, x2), max_x = Math.max(x1, x2),
            min_y = Math.min(y1, y2), max_y = Math.max(y1, y2);
        var new_area = {
            x: (min_x - plot_x) / plot_w,
            y: (min_y - plot_y) / plot_h,
            w: (max_x - min_x) / plot_w,
            h: (max_y - min_y) / plot_h
        };
        if (new_area.x + new_area.w > 1 && new_area.x < 1) {
            return relative_sampling_area;
        } else {
            return new_area;
        }
    }

    that.get_sampling_area = function() {
        return JSON.parse(JSON.stringify(sampling_area));
    };
    that.set_sampling_area = function(area) {
        sampling_area = area;
    };
    that.reset_sampling_area = function() {
        that.set_sampling_area({
            x: 0,
            y: 0,
            w: 1,
            h: 1
        });
        data_id = -1;
        that.resample();
    };
    that.resample = function() {
        container.select("#viewbox").style("visibility", "hidden");
        container.select("#confirm-button").style("visibility", "hidden");
        let datatype = "";
        if (parent.get_position_mode() !== "juxtaposition") {
            datatype = parent.get_data_type();
        } else {
            if (relative_sampling_area.x > 1) {
                datatype = "test";
                relative_sampling_area.x = relative_sampling_area.x - small_x_2 / plot_width;
            } else {
                datatype = "train";
            }
        }
        if (datatype === "train") {
            data_id = train_id;
        } else if (datatype === "test") {
            data_id = test_id;
        } else {
            data_id = all_id;
        }
        console.log('relative_sampling_area');
        var category_code = LabelShown.reduce((str, d) => str + (d ? "1" :"0"), "");
        d3.select("#preloader").style("display", "block");
        Loader.grid_data_node.set_url(
            GridLayoutApi + "?dataset=" + Loader.dataset + "&datatype=" + datatype + "&embed-method=tsne" +
            "&left-x=" + (relative_sampling_area.x) +
            "&top-y=" + (relative_sampling_area.y) +
            "&width=" + (relative_sampling_area.w) +
            "&height=" + (relative_sampling_area.h) +
            "&distribution=" + datatype +
            "&node-id=" + data_id +
            "&class=" + category_code
        );
        Loader.grid_data_node.set_on();
    };

    that.zoomed = function (x, y, k){
        let t = d3.zoomIdentity;
        t = t.translate(x, y).scale(k);
        group_container.attr("transform",t);
    };

    that.create = function(){
        // that.boundingbox_create();
        // that.pattern_create();
        // that.image_create();
    };
    that.update = function(){
        console.log("distribution update");
        var grids = plot_group.selectAll("g.grid").data(all_data, d => d.get_id());
        var enter_size = grids.enter().size(),
            update_size = grids.size(),
            exit_size = grids.exit().size();
        var cumulated_animation_time = 0,
            single_animation_time = 1000;

        var patterns = pattern_group.selectAll("pattern").data(all_data, d => d.get_id());

        // Animation
        if (exit_size > 0) {
            patterns.exit()
                .transition()
                .duration(single_animation_time)
                .remove();
            grids.exit()
                .transition()
                .duration(single_animation_time)
                .style("opacity", 0)
                .remove();
            cumulated_animation_time += single_animation_time;
        }
        if (update_size > 0) {
            patterns.attr("id", d => "radius-img-" + d.get_id())
                .transition()
                .duration(single_animation_time)
                .delay(cumulated_animation_time)
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width)
                .select("image")
                // .attr("xlink:href", function(d){
                //     let width = d.width * plot_width - boundingbox_width;
                //     if (width < 30){
                //         return d.get_thumbnail_url();
                //     }
                //     else{
                //         return d.get_img_url();
                //     }
                // })
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width);
            grids.attr("id", d => "ID-" + d.get_id())
                .each(function(d) {
                   d3.select(this)
                        .select(".boundingbox")
                        .transition()
                        .duration(single_animation_time)
                        .delay(cumulated_animation_time)
                        .attr("width", d.width * plot_width)
                        .attr("height", d.width * plot_width)
                        .attr("rx", d.width * plot_width / 2 * (d.get_datatype() === "train"))
                        .attr("ry", d.width * plot_width / 2 * (d.get_datatype() === "train"));
                   d3.select(this)
                        .select(".display")
                        .transition()
                        .duration(single_animation_time)
                        .delay(cumulated_animation_time)
                        .attr("width", d.width * plot_width - boundingbox_width)
                        .attr("height", d.width * plot_height - boundingbox_width)
                        .attr("rx", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .attr("ry", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .attr("fill", "url(#radius-img-" + d.get_id() + ")");
                })
                .transition()
                .duration(single_animation_time)
                .delay(cumulated_animation_time)
                .attr("transform", d => "translate(" + (d.grid_x[0] * plot_width) + ", " +  (d.grid_x[1] * plot_width) + ")")
                .attr("width", d => d.width * plot_width)
                .attr("height", d => d.width * plot_width);
            cumulated_animation_time += single_animation_time;
        }
        if (enter_size > 0) {
            patterns.enter()
                .append("pattern")
                .attr("id", d => "radius-img-" + d.get_id())
                .attr("patternUnits", "userSpaceOnUse")
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width)
                .append("image")
                // .attr("xlink:href", function(d){
                //     let width = d.width * plot_width - boundingbox_width;
                //     if (width < 50){
                //         return d.get_thumbnail_url();
                //     }
                //     else{
                //         return d.get_img_url();
                //     }
                // })
                .attr("xlink:href", d => d.get_img_url())
                .attr("width", d => d.width * plot_width - boundingbox_width)
                .attr("height", d => d.width * plot_height - boundingbox_width);
            grids.enter()
                .append("g")
                .attr("id", d => "ID-" + d.get_id())
                .attr("class", "grid")
                .attr("transform", d => "translate(" + (d.grid_x[0] * plot_width) + ", " +  (d.grid_x[1] * plot_width) + ")")
                .attr("width", d => d.width * plot_width)
                .attr("height", d => d.width * plot_width)
                .style("opacity", 0)
                .on('click', function(d) {
                    Loader.focus_data_node.set_url(FocusApi + "?dataset=" + Loader.dataset + "&id=" + d.get_id() + "&k=" + 9);
                    Loader.focus_data_node.set_handler(data => focus_handler(parent, data));
                    Loader.focus_data_node.set_on();
                    d3.select("#info-panel")
                        .style("display", "block")
                        .style("left", d3.event.pageX)
                        .style("top", d3.event.pageY);
                })
                .on("mousemove", function() {
                    d3.select(this).select("rect")
                        .style("stroke-width",  2.0);
                })
                .on("mouseout", function() {
                    d3.select(this).select("rect")
                        .style("stroke-width", 0.0);
                })
                .on("mouseenter", function(d){
                    d3.select(this).select("rect")
                        .style("stroke-width",  2.0);
                    parent.tips_in_another(d.get_feature(), true);
                })
                .on("mouseleave", function(d){
                    d3.select(this).select("rect")
                        .style("stroke-width",  0.0);
                    parent.tips_in_another(d.get_feature(), false);
                })
                .each(function(d) {
                    d3.select(this)
                        .append("rect")
                        .attr("class", "boundingbox")
                        .attr("x", 0)
                        .attr("y", 0)
                        .attr("width", d.width * plot_width)
                        .attr("height", d.width * plot_width)
                        .attr("rx", d.width * plot_width / 2 * (d.get_datatype() === "train"))
                        .attr("ry", d.width * plot_width / 2 * (d.get_datatype() === "train"))
                        .style("fill", CategoryColor[d.get_y()])
                        .style("stroke", "black")
                        .style("stroke-width", 0)
                        .style("stroke-opacity", 1);
                    d3.select(this)
                        .append("rect")
                        .attr("class", "display")
                        .attr("x", 0.5 * boundingbox_width)
                        .attr("y", 0.5 * boundingbox_width)
                        .attr("width", d.width * plot_width - boundingbox_width)
                        .attr("height", d.width * plot_height - boundingbox_width)
                        .attr("rx", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .attr("ry", (d.width * plot_width - boundingbox_width) / 2 * (d.get_datatype() === "train"))
                        .attr("fill", "url(#radius-img-" + d.get_id() + ")")
                        .style("pointer-events", "none");

                })
                .transition()
                .duration(single_animation_time)
                .delay(cumulated_animation_time)
                .style("opacity", 1);
        }

    };
    that.remove = function(){
        // that.boundingbox_remove();
        // that.pattern_remove();
        // that.image_remove();
    };

    that.layout = function(data, plot_size, boundary_points) {
        let padding_label = 40;
        if (data.length <= 25**2) {
            labels = Array();
            return null;
        }
        var grid_N = Math.ceil(Math.sqrt(data.length));
        var grid_size = plot_size / grid_N;
        var img_size = img_width;
        var new_labels = Array();
        var restrict_points = boundary_points.map(p => p.map(d => d * plot_size + 0.5 * grid_size));
        var inside = function(point, rect) {
            return !((rect.x + rect.w < point[0] - 0.5 * grid_size) || (rect.x > point[0] + 0.5 * grid_size) ||
                (rect.y + rect.h < point[1] - 0.5 * grid_size) || (rect.y > point[1] + 0.5 * grid_size))
        };
        var intersect = function(rect1, rect2) {
            return !((rect1.x + rect1.w + padding_label < rect2.x) || (rect2.x + rect2.w + padding_label < rect1.x) ||
                (rect1.y + rect1.h + padding_label < rect2.y) || (rect2.y + rect2.h + padding_label < rect1.y))
        };
        var legal = function(rect) {
            return rect.x > 0 &&
                rect.y > 0 &&
                rect.x + rect.w < plot_size &&
                rect.y + rect.h < plot_size;
        };
        const offset = [
            { x: 0, y: 0 },
            { x: 0, y: -img_size },
            { x: -img_size, y: -img_size },
            { x: -img_size, y: 0 }
        ];
        data.sort((x, y) => y.get_entropy() - x.get_entropy());
        for (let d of data) {
            var center = {
                x: (d.coord[0] + 0.5) * grid_size,
                y: (d.coord[1] + 0.5) * grid_size,
            };
            var tmp_grid = {
                x: d.grid_x[0] * plot_size,
                y: d.grid_x[1] * plot_size,
                w: d.width * plot_size,
                h: d.width * plot_size
            };
            var tmp_label;
            var can_placed;
            var prev_label = labels.filter(lbl => lbl.get_id() === d.get_id());
            if (prev_label.length > 0) {
                var direction = prev_label[0].label.dir;
                tmp_label = {
                    dir: direction,
                    x: center.x + offset[direction].x,
                    y: center.y + offset[direction].y,
                    w: img_size,
                    h: img_size
                };
                can_placed = true;
                for (let r of new_labels) {
                    if (intersect(tmp_label, r.label) || intersect(tmp_grid, r.label) ||
                        intersect(tmp_label, r.grid) || intersect(tmp_grid, r.grid) ||
                        !legal(tmp_label)) {
                        can_placed = false;
                        break;
                    }
                }
                if (can_placed) {
                    for (let p of restrict_points) {
                        if (inside(p, tmp_label)) {
                            can_placed = false;
                            break;
                        }
                    }
                }
                if (can_placed) {
                    new_labels.push({
                        label: tmp_label,
                        grid: tmp_grid,
                        ...d
                    });
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
                    for (let r of new_labels) {
                        if (intersect(tmp_label, r.label) || intersect(tmp_grid, r.label) ||
                            intersect(tmp_label, r.grid) || intersect(tmp_grid, r.grid) ||
                            !legal(tmp_label)) {
                            can_placed = false;
                            break;
                        }
                    }
                    if (can_placed) {
                        for (let p of restrict_points) {
                            if (inside(p, tmp_label)) {
                                can_placed = false;
                                break;
                            }
                        }
                    }
                    if (can_placed) {
                        new_labels.push({
                            label: tmp_label,
                            grid: tmp_grid,
                            ...d
                        });
                        break;
                    }
                }
            }
        }
        return new_labels;
};


    that.tips_in_another = function(f, tip_type, data_type) {
        var dist = function(x, y) {
            var len = x.length;
            var sum = 0;
            for (var i = 0; i < len; ++i) {
                sum += (x[i] - y[i])**2;
            }
            return Math.sqrt(sum);
        };
        let another;
        if (data_type === "training"){
            another = that.test_view;
        }
        else{
            another = that.training_view;
        }
        if (parent.get_position_mode() === "juxtaposition"){
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

    that.setImageVisible = function(visible){
        image_visible = visible;
        that.training_view.setImagevisible(visible);
        that.test_view.setImagevisible(visible);
    };

    that.enter_overview = function() {
        if (parent.get_position_mode() !== "juxtaposition") {
            overview_group.select("#overview-1")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", large_grid_width)
                .attr("height", large_grid_width);
            overview_group.select("#overview-2")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", 0)
                .attr("height", 0);
        } else {
            overview_group.select("#overview-1")
                .attr("x", small_x_1)
                .attr("y", small_y_1)
                .attr("width", small_grid_width)
                .attr("height", small_grid_width);
            overview_group.select("#overview-2")
                .attr("x", small_x_2)
                .attr("y", small_y_2)
                .attr("width", small_grid_width)
                .attr("height", small_grid_width);
        }
        // that.training_view.setImagevisible("hidden");
        // that.test_view.setImagevisible("hidden");
        overview_group.style("visibility", "visible");
        group_container.select("#confirm-resample").style("visibility", "hidden");
    };
    that.quit_overview = function() {
        // that.training_view.setImagevisible(image_visible);
        // that.test_view.setImagevisible(image_visible);
        overview_group.style("visibility", "hidden");
        container.select("#viewbox").style("visibility", "hidden");
        container.select("#confirm-resample").style("visibility", "hidden");
        for (let i = 0; i < train_data.length; i++){
            let data_item = train_data[i];
            let item = train_data_svg_items[i];

            data_item.selected = false;
            if(data_item.color !== undefined){
                item.style('fill', data_item.color);
            }
        }
        for (let i = 0; i < test_data.length; i++){
            let data_item = test_data[i];
            let item = test_data_svg_items[i];

            data_item.selected = false;
            if(data_item.color !== undefined){
                item.style('fill', data_item.color);
            }
        }
    };
};
