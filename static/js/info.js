/**
 * Created by Changjian on 2018/12/17.
 */

var InfoLayout = function(container) {
    var that = this;
    that.container = container;

    let bbox = document.getElementById("info-panel").getBoundingClientRect();
    let layout_width = bbox.width;
    let layout_height = bbox.height * 0.45;
    let img_offset_x = 20;
    let img_offset_y = 10;
    let img_padding = 10;
    let img_width = (layout_width - 2 * img_padding - 2 * img_offset_x) / 3;
    let img_height = img_width;
    let text_offset_x = 20;
    let text_offset_y = img_height + img_offset_y + 30 ;
    let text_height = 30;
    let boundingbox_width = 0.05 * img_width;
    let button_size = 10;

    var mouse_pressed = false;
    var mouse_pos = {
        x: -1,
        y: -1
    };
    var info_svg_offset_y = 0;

    let saliency_mode = true;
    let neighbour_num = 5;
    let image_url = null;
    let all_data = Array();
    let neighbour_data = Array();
    let info_images = null, neighbour_images = null, info_patterns = null, neighbour_patterns = null;

    var info_svg = that.container.select("#block-info").append("svg").attr("id", "info-svg");
    var info_img_group = info_svg.append("g").attr("id", "info-img-group");
    var info_pattern_group = info_svg.append("defs").attr("id", "info-pattern-group");
    // var text_group = svg.append("g").attr("id", "text-group");
    var neighbour_svg = that.container.select("#block-neighbour").append("svg").attr("id", "neighbour-svg");
    var neighbour_img_group = neighbour_svg.append("g").attr("id", "neighbour-img-group");
    var neighbour_pattern_group = neighbour_svg.append("defs").attr("id", "neighbour-pattern-group");

    var go_up = that.container.select("#block-info").append("svg").attr("id", "info-go-up").attr("viewBox", "0 0 1024 1024");
    var go_down = that.container.select("#block-info").append("svg").attr("id", "info-go-down").attr("viewBox", "0 0 1024 1024");

    var detail_group = info_svg.append("g").attr("id", "info-detail");
    var detail_pos = -1;

    // data
    var id = null;
    var dc_distance = null;
    var entropy = null;
    var classification_result = null;

    that._init = function(){
        // set up svg's width and height
        info_svg.attr("width", layout_width)
            .attr("height", layout_height);
        $(info_svg.node())[0].onmousewheel = function(event) {
            event = event || window.event;
            if (Math.ceil(all_data.length / 3) * (img_width + img_padding) > layout_height) {
                info_svg_offset_y += event.wheelDelta / 30;
                if (info_svg_offset_y > 0) {
                    info_svg_offset_y = 0;
                } else if (info_svg_offset_y + Math.ceil(all_data.length / 3) * (img_width + img_padding) < layout_height - img_padding) {
                    info_svg_offset_y = layout_height - img_padding - Math.ceil(all_data.length / 3) * (img_width + img_padding);
                }
                info_img_group.attr("transform", "translate(20, " + (info_svg_offset_y + 10) + ")");
                detail_group.attr("transform", "translate(20, " + (info_svg_offset_y + 10) + ")");
            }
        };
        info_img_group.attr("transform", "translate(" + ( img_offset_x ) + "," + ( img_offset_y )+ ")");
        detail_group.attr("transform", "translate(" + ( img_offset_x ) + "," + ( img_offset_y )+ ")");
        detail_group.append("rect")
            .attr("id", "detail-back")
            .attr("width", layout_width - 2 * img_offset_x)
            .attr("height", 0)
            .style("fill", "#42403E");
        detail_group.append("polygon")
            .attr("id", "indicator")
            .attr("points", "0,0")
            .style("fill", "#42403E");
        detail_group.append("image");
        neighbour_svg.attr("width", layout_width)
            .attr("height", layout_height);
        neighbour_img_group.attr("transform", "translate(" + ( img_offset_x ) + "," + ( img_offset_y )+ ")");

        go_up.attr("transform", "translate(" + (layout_width - 2 * button_size) + ", " +
            (- 6 * button_size) + ")")
            .attr("width", 2 * button_size)
            .attr("height", 2 * button_size)
            .attr("class", "info-wheel")
            .style("position", "absolute");
        go_up.append("path")
            .attr("d", "M224.492308 425.353846L490.338462 155.569231c11.815385-11.815385 31.507692-11.815385 43.323076 0l265.846154 269.784615c11.815385 11.815385 11.815385 31.507692 0 43.323077L756.184615 512c-11.815385 11.815385-31.507692 11.815385-43.323077 0l-179.2-185.107692c-11.815385-11.815385-31.507692-11.815385-43.323076 0l-179.2 183.138461c-11.815385 11.815385-31.507692 11.815385-43.323077 0l-43.323077-43.323077c-9.846154-11.815385-9.846154-29.538462 0-41.353846z m0 356.430769L490.338462 512c11.815385-11.815385 31.507692-11.815385 43.323076 0l265.846154 269.784615c11.815385 11.815385 11.815385 31.507692 0 43.323077l-43.323077 43.323077c-11.815385 11.815385-31.507692 11.815385-43.323077 0l-179.2-185.107692c-11.815385-11.815385-31.507692-11.815385-43.323076 0L311.138462 866.461538c-11.815385 11.815385-31.507692 11.815385-43.323077 0l-43.323077-43.323076c-9.846154-11.815385-9.846154-31.507692 0-41.353847z")
            .attr("fill", "#333333");
        go_down.attr("transform", "translate(" + (layout_width - 2 * button_size) + ", " +
            (- 3 * button_size) + ")")
            .attr("width", 2 * button_size)
            .attr("height", 2 * button_size)
            .attr("class", "info-wheel")
            .style("position", "absolute");
        go_down.append("path")
            .attr("d", "M799.507692 598.646154L533.661538 866.461538c-11.815385 11.815385-31.507692 11.815385-43.323076 0L224.492308 598.646154c-11.815385-11.815385-11.815385-31.507692 0-43.323077l43.323077-43.323077c11.815385-11.815385 31.507692-11.815385 43.323077 0l179.2 185.107692c11.815385 11.815385 31.507692 11.815385 43.323076 0l179.2-183.138461c11.815385-11.815385 31.507692-11.815385 43.323077 0l43.323077 43.323077c9.846154 11.815385 9.846154 29.538462 0 41.353846z m0-356.430769L533.661538 513.969231c-11.815385 11.815385-31.507692 11.815385-43.323076 0L224.492308 242.215385c-11.815385-11.815385-11.815385-31.507692 0-43.323077l43.323077-43.323077c11.815385-11.815385 31.507692-11.815385 43.323077 0l179.2 185.107692c11.815385 11.815385 31.507692 11.815385 43.323076 0L712.861538 157.538462c11.815385-11.815385 31.507692-11.815385 43.323077 0l43.323077 43.323076c9.846154 11.815385 9.846154 31.507692 0 41.353847z")
            .attr("fill", "#333333");

        d3.selectAll(".info-wheel")
            .on("mousedown", function() {
                var offset_delta = (img_width + img_padding) * (d3.select(this).attr("id") === "info-go-up" ? 1 : -1);
                if (Math.ceil(all_data.length / 3) * (img_width + img_padding) > layout_height) {
                    info_svg_offset_y += offset_delta;
                    if (info_svg_offset_y > 0) {
                        info_svg_offset_y = 0;
                    } else if (info_svg_offset_y + Math.ceil(all_data.length / 3) * (img_width + img_padding) < layout_height - img_padding) {
                        info_svg_offset_y = layout_height - img_padding - Math.ceil(all_data.length / 3) * (img_width + img_padding);
                    }
                    info_img_group.attr("transform", "translate(20, " + (info_svg_offset_y + 10) + ")");
                    detail_group.attr("transform", "translate(20, " + (info_svg_offset_y + 10) + ")");
                }
            })

        // text_group.attr("transform",
        //     "translate(" + ( text_offset_x ) + "," + ( text_offset_y )+ ")");

        // d3.select("#info-panel")
        //     .on("mousedown", function() {
        //         mouse_pressed = true;
        //         mouse_pos = {
        //             x: d3.event.pageX,
        //             y: d3.event.pageY
        //         }
        //     });
        // var body = $("body")[0];
        // body.addEventListener("mousemove", function(event) {
        //     if (!mouse_pressed) {
        //         return;
        //     }
        //     var x = d3.select("#info-panel").style("left"),
        //         y = d3.select("#info-panel").style("top");
        //     x = Number(x.substring(0, x.length - 2));
        //     y = Number(y.substring(0, y.length - 2));
        //     d3.select("#info-panel")
        //         .style("left", x + event.pageX - mouse_pos.x)
        //         .style("top", y + event.pageY - mouse_pos.y);
        //     mouse_pos = {
        //         x: event.pageX,
        //         y: event.pageY
        //     }
        // });
        // body.addEventListener("mouseup", function() {
        //     mouse_pressed = false;
        // });
        // d3.select("#close-info")
        //     .on("click", function() {
        //         d3.select("#info-panel").style("display", "none");
        //     })
        };

    that.init = function(){
        that._init();
    }.call();

    that.draw_images = function() {
        info_images = info_img_group.selectAll("g.info").data(all_data, d => d.get_id());
        info_patterns = info_pattern_group.selectAll("pattern.info").data(all_data, d => d.get_id());
        that.image_create();
        that.image_update();
        that.image_remove();
    };

    that.draw_neighbours = function() {
        neighbour_images = neighbour_img_group.selectAll("g.neighbour").data(neighbour_data, d => d.get_id());
        neighbour_patterns = neighbour_pattern_group.selectAll("pattern.neighbour").data(neighbour_data, d => d.get_id());
        that.neighbour_create();
        that.neighbour_update();
        that.neighbour_remove();
    };

    that.image_create = function(){
        info_images.enter()
            .append("g")
            .attr("class", "info")
            .attr("width", img_width)
            .attr("height", img_height)
            .attr("transform", (d, i) => "translate(" + ((i % 3) * (img_width + img_padding)) + ", " +
                (parseInt(i / 3) * (img_width + img_padding)) + ")")
            .style("opacity", 0)
            .on("click", (d, i) => {
                // Neighbour
                Loader.focus_data_node.set_url(FocusApi + "?dataset=" + Loader.dataset + "&id=" + d.get_id() + "&k=" + 9);
                Loader.focus_data_node.set_on();
                // Magnify
                if (detail_pos === -1) {
                    detail_pos = i;
                    detail_group.transition()
                        .duration(AnimationDuration)
                        .style("opacity", 1);
                    detail_group.select("#detail-back")
                        .attr("y", ((parseInt(i / 3) + 1) * (img_width + img_padding)))
                        .transition()
                        .duration(AnimationDuration)
                        .style("height", 2 * (img_width + img_padding));
                    detail_group.select("image")
                        .attr("xlink:href", d.get_img_url())
                        .attr("x", layout_width / 2 - img_offset_x)
                        .attr("y", ((parseInt(i / 3) + 1) * (img_width + img_padding)) + img_padding)
                        .attr("width", 0)
                        .attr("height", 0)
                        .transition()
                        .duration(AnimationDuration)
                        .attr("x", layout_width / 2 - img_width - img_offset_x)
                        .attr("y", ((parseInt(i / 3) + 1) * (img_width + img_padding)) + img_padding)
                        .attr("width", img_width * 2)
                        .attr("height", img_width * 2);
                    detail_group.select("polygon")
                        .attr("points", (img_width / 2 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding) - img_padding))
                        .transition()
                        .duration(AnimationDuration)
                        .attr("points", (img_width / 2 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding) - img_padding) + " "
                            + (img_width / 2 - 5 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding)) + " "
                            + (img_width / 2 + 5 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding)));
                    that.draw_images();
                } else if (detail_pos === i) {
                    detail_pos = -1;
                    detail_group.transition()
                        .duration(AnimationDuration)
                        .style("opacity", 0);
                    detail_group.select("#detail-back")
                        .attr("y", ((parseInt(i / 3) + 1) * (img_width + img_padding)))
                        .transition()
                        .duration(AnimationDuration)
                        .style("height", 0);
                    detail_group.select("image")
                        .transition()
                        .duration(AnimationDuration)
                        .attr("x", layout_width / 2 - img_offset_x)
                        .attr("y", ((parseInt(i / 3) + 1) * (img_width + img_padding)) + img_padding)
                        .attr("width", 0)
                        .attr("height", 0);
                    detail_group.select("polygon")
                        .transition()
                        .duration(AnimationDuration)
                        .attr("points", (img_width / 2 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding) - img_padding));
                    that.draw_images();
                } else {
                    detail_pos = i;
                    detail_group.transition()
                        .duration(AnimationDuration)
                        .style("opacity", 1);
                    detail_group.select("#detail-back")
                        .transition()
                        .duration(AnimationDuration)
                        .attr("y", ((parseInt(i / 3) + 1) * (img_width + img_padding)));
                    detail_group.select("image")
                        .attr("xlink:href", d.get_img_url())
                        .transition()
                        .duration(AnimationDuration)
                        .attr("x", layout_width / 2 - img_width - img_offset_x)
                        .attr("y", ((parseInt(i / 3) + 1) * (img_width + img_padding)) + img_padding)
                        .attr("width", img_width * 2)
                        .attr("height", img_width * 2);
                    detail_group.select("polygon")
                        .transition()
                        .duration(AnimationDuration)
                        .attr("points", (img_width / 2 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding) - img_padding) + " "
                            + (img_width / 2 - 5 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding)) + " "
                            + (img_width / 2 + 5 + (i % 3) * (img_padding + img_width)) + "," + ((parseInt(i / 3) + 1) * (img_width + img_padding)));
                    that.draw_images();
                }
            })
            .each(function(d) {
                var g = d3.select(this);
                g.append("rect")
                    .attr("class", "bbox-info")
                    .attr("x", 0)
                    .attr("y", 0)
                    .attr("width", img_width)
                    .attr("height", img_height)
                    .attr("rx", img_width / 2 * (d.get_datatype() === "train"))
                    .attr("ry", img_height / 2 * (d.get_datatype() === "train"))
                    .attr("fill", CategoryColor[d.get_y()]);
                g.append("rect")
                    .attr("class", "img-info")
                    .attr("x", boundingbox_width)
                    .attr("y", boundingbox_width)
                    .attr("width", img_width - 2 * boundingbox_width)
                    .attr("height", img_height - 2 * boundingbox_width)
                    .attr("rx", (img_width / 2 - boundingbox_width) * (d.get_datatype() === "train"))
                    .attr("ry", (img_height / 2 - boundingbox_width) * (d.get_datatype() === "train"))
                    .attr("fill", "url(#info-img-" + d.get_id() + ")");
            })
            .transition()
            .duration(AnimationDuration)
            .style("opacity", 1);
        info_patterns.enter()
            .append("pattern")
            .attr("class", "info")
            .attr("id", d => "info-img-" + d.get_id())
            .attr("patternUnits", "userSpaceOnUse")
            .attr("x", 0.5 * boundingbox_width)
            .attr("y", 0.5 * boundingbox_width)
            .attr("width", img_width - boundingbox_width)
            .attr("height", img_height - boundingbox_width)
            .append("image")
            .attr("xlink:href", d => saliency_mode ? d.get_saliency_map_url() : d.get_img_url())
            .attr("width", img_width - boundingbox_width)
            .attr("height", img_height - boundingbox_width);
    };
    that.image_update = function(){
        info_images.transition()
            .duration(AnimationDuration)
            .attr("transform", (d, i) => "translate(" + ((i % 3) * (img_width + img_padding)) + ", " +
                (parseInt(i / 3) * (img_width + img_padding) + (detail_pos !== -1 && parseInt(i / 3) > parseInt(detail_pos / 3)) * (img_width * 2 + img_padding * 3)) + ")");
        info_patterns.select("image")
            .attr("xlink:href", d => saliency_mode ? d.get_saliency_map_url() : d.get_img_url());
    };
    that.image_remove = function(){
        info_patterns.exit()
            .transition()
            .duration(AnimationDuration)
            .remove();
        info_images.exit()
            .transition()
            .duration(AnimationDuration)
            .style("opacity", 0)
            .remove();
    };

    that.neighbour_create = function() {
        neighbour_images.enter()
            .append("g")
            .attr("class", "neighbour")
            .attr("width", img_width)
            .attr("height", img_height)
            .attr("transform", (d, i) => "translate(" + ((i % 3) * (img_width + img_padding)) + ", " +
                (parseInt(i / 3) * (img_width + img_padding)) + ")")
            .style("opacity", 0)
            .each(function(d) {
                var g = d3.select(this);
                g.append("rect")
                    .attr("class", "bbox-neighbour")
                    .attr("x", 0)
                    .attr("y", 0)
                    .attr("width", img_width)
                    .attr("height", img_height)
                    .attr("rx", img_width / 2 * (d.get_datatype() === "train"))
                    .attr("ry", img_height / 2 * (d.get_datatype() === "train"))
                    .attr("fill", CategoryColor[d.get_y()]);
                g.append("rect")
                    .attr("class", "img-neighbour")
                    .attr("x", boundingbox_width)
                    .attr("y", boundingbox_width)
                    .attr("width", img_width - 2 * boundingbox_width)
                    .attr("height", img_height - 2 * boundingbox_width)
                    .attr("rx", (img_width / 2 - boundingbox_width) * (d.get_datatype() === "train"))
                    .attr("ry", (img_height / 2 - boundingbox_width) * (d.get_datatype() === "train"))
                    .attr("fill", "url(#neighbour-img-" + d.get_id() + ")");
            })
            .transition()
            .duration(AnimationDuration)
            .style("opacity", (d, i) => i < neighbour_num ? 1 : 0);
        neighbour_patterns.enter()
            .append("pattern")
            .attr("class", "neighbour")
            .attr("id", d => "neighbour-img-" + d.get_id())
            .attr("patternUnits", "userSpaceOnUse")
            .attr("x", 0.5 * boundingbox_width)
            .attr("y", 0.5 * boundingbox_width)
            .attr("width", img_width - boundingbox_width)
            .attr("height", img_height - boundingbox_width)
            .append("image")
            .attr("xlink:href", d => saliency_mode ? d.get_saliency_map_url() : d.get_img_url())
            .attr("width", img_width - boundingbox_width)
            .attr("height", img_height - boundingbox_width);

    };
    that.neighbour_update = function() {
        neighbour_images.transition()
            .duration(AnimationDuration)
            .attr("transform", (d, i) => "translate(" + ((i % 3) * (img_width + img_padding)) + ", " +
                (parseInt(i / 3) * (img_width + img_padding)) + ")")
            .style("opacity", (d, i) => i < neighbour_num ? 1 : 0);
        neighbour_patterns.select("image")
            .attr("xlink:href", d => saliency_mode ? d.get_saliency_map_url() : d.get_img_url());
    };
    that.neighbour_remove = function() {
        neighbour_patterns.exit()
            .transition()
            .duration(AnimationDuration)
            .remove();
        neighbour_images.exit()
            .transition()
            .duration(AnimationDuration)
            .style("opacity", 0)
            .remove();
    };

    that.text_create = function(){
        let data = [id, entropy, dc_distance];
        text_group.selectAll("text.info-text")
            .data(data)
            .enter()
            .append("text")
            .attr("class", "info-text");
    };
    that.text_update = function(){
        let data = [id, entropy, dc_distance];
        text_group.selectAll("text.info-text")
            .data(data)
            .attr("x", function(d,i){
                return 0;
            })
            .attr("y", function(d, i){
                return i * text_height;
            })
            .text(function(d){
                return d;
            });
    };
    that.text_remove = function(){
        let data = [id, entropy, dc_distance];
        text_group.selectAll("div.info-text")
            .data(data)
            .exit()
            .transition()
            .duration(AnimationDuration)
            .attr("opacity", 0)
            .remove();
    };

    that.update_info = function(infos){
        image_url = SaliencyMapApi + "?dataset="
            + DatasetName + "&filename=" + infos["id"] + ".jpg";
        id = "id: " + infos["id"];
        entropy = "entropy: " +  infos["ent"];
        dc_distance = "ground truth: " + infos["gt"];
        console.log(image_url);
        that.draw();
    };

    that.load_data = function(data) {
        console.log(data);
        data.sort();
        info_svg_offset_y = 0;
        detail_pos = -1;
        info_img_group.attr("transform", "translate(20, " + (info_svg_offset_y + 10) + ")");
        detail_group.attr("transform", "translate(20, " + (info_svg_offset_y + 10) + ")");
        detail_pos = -1;
        detail_group.style("opacity", 0);
        all_data = data.map(d => Loader.TrainData.get_cell_by_id(d) || Loader.TestData.get_cell_by_id(d));
        that.update_neighbours([]);
    };

    that.switch_display_images = function(open_saliency) {
        saliency_mode = open_saliency;
        that.draw_images();
        that.draw_neighbours();
    };

    that.update_neighbour_num = function(k) {
        neighbour_num = k;
        that.draw_neighbours();
    };

    that.update_neighbours = function(data) {
        neighbour_data = data.map(d => Loader.TrainData.get_cell_by_id(d) || Loader.TestData.get_cell_by_id(d));
        that.draw_neighbours();
    };

    that.open = function() {
        d3.select("#info-panel").style("display", "block");
    };

    that.close = function() {
        d3.select("#info-panel").style("display", "none");
    };
};