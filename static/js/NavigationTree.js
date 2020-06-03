var NavigationTree = function(container) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var layout_width = bbox.width;
    var layout_height = bbox.height - 28;
    var margin = {top: 20, right: 0, bottom: 0, left: 0};
    var rect_height = 10;
    var menu_width = 70, menu_height = 30;
    var font_size = 15;

    var duration = 500;

    var svg = that.container.append("svg").attr("id", "navigation");
    var group_container = svg.append("g").attr("id", "nav-tree");
    var menu = svg.append("g").attr("id", "nav-menu");

    // data
    var train_nav = {root: null};
    var test_nav = {root: null};
    var all_nav = {root: null};
    var id_set = new Set([]);
    var current_status = null;
    var current_datatype = "test";
    var selected_status = null;
    var root = null;
    var treemap = d3.tree().size([layout_width, layout_height]);
    var treeData;

    var another = null;

    function compute_encoding(cells) {
        var encoding = Array(LabelNames.length).fill(0);
        cells.forEach(d => {
            encoding[d.get_y()] += 1
        });
        return encoding;
    }

    // function collapse(d) {
    //     if (d.children.length > 0) {
    //         d._children = d.children;
    //         d._children.forEach(collapse);
    //         d.children = null;
    //     }
    // }

    function update(source) {
        var nodes = treeData.descendants(),
            links = treeData.descendants().slice(1);
        nodes.forEach(d => {
            d.y = d.depth * 50;
            d.width = Math.sqrt(d.data.node.nb_instances) * 3;
        });
        var node = group_container.selectAll("g.tree-node")
            .data(nodes, d => d.data.id);
        var nodeEnter = node.enter()
            .append("g")
            .attr("id", d => "nav-node-" + d.data.id)
            .attr("class", "tree-node")
            .attr("transform", "translate(" + source.x0 + ", " + source.y0 + ")")
            .attr("cursor", "pointer")
            .on("mousedown", d => {
                selected_status = d.data;
                if (d3.event.button === 0) {
                    reload_status();
                } else if (d3.event.button === 2) {
                    var offset = $(svg.node()).offset();
                    var x = d3.event.pageX, y = d3.event.pageY;
                    var check_del_valid, s = current_status;
                    while (s.parent !== null && selected_status !== s) {
                        s = s.parent;
                    }
                    if (s === selected_status) {
                        check_del_valid = false;
                    } else {
                        check_del_valid = true;
                    }
                    menu.style("visibility", "visible")
                        .attr("transform", "translate(" + (x - offset.left) + ", " + (y - offset.top) + ")");
                    menu.select("#nav-reload")
                        .style("visibility", "visible");
                    menu.select("#nav-delete")
                        .style("visibility", check_del_valid ? "visible" : "hidden");
                    // menu.select("#nav-collapse")
                    //     .style("visibility", check_del_valid ? "visible" : "hidden");
                }
            });
        nodeEnter.append("rect")
            .attr("class", "node")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 0)
            .attr("height", 0)
            .style("fill", "white")
            .style("stroke", "grey")
            .style("stroke-width", 0);
        nodeEnter.each(function(d) {
            d3.select(this)
                .selectAll("rect.encoding")
                .data(d.data.node.encoding)
                .enter()
                .append("rect")
                .attr("class", "encoding")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", 0)
                .attr("height", 0)
                .attr("fill", (r, i) => CategoryColor[i]);
        });
        var nodeUpdate = nodeEnter.merge(node);
        nodeUpdate.transition()
            .duration(duration)
            .attr("transform", d => "translate(" + d.x + ", " + d.y + ")");
        nodeUpdate.each(function(d) {
                var encoding = d.data.node.encoding;
                d3.select(this)
                    .selectAll("rect.encoding")
                    .transition()
                    .duration(duration)
                    .attr("x", (r, i) => -d.width / 2 + encoding.slice(0, i).reduce((a, b) => a + b, 0) / d.data.node.nb_instances * d.width)
                    .attr("y", -rect_height / 2)
                    .attr("width", r => r / d.data.node.nb_instances * d.width)
                    .attr("height", rect_height);
            })
            .select("rect.node")
            .attr("class", "node")
            .attr("x", d => -d.width / 2)
            .attr("y", -rect_height / 2)
            .attr("width", d => d.width)
            .attr("height", rect_height)
            .style("stroke", d => d.data.id === current_status.id ? "black" : "grey")
            .style("stroke-width", 5);
        var nodeExit = node.exit()
            .transition()
            .duration(duration)
            .attr("transform", "translate(" + source.x + "," + source.y + ")")
            .remove();
        nodeExit.each(function() {
                d3.select(this)
                    .selectAll("rect.encoding")
                    .transition()
                    .duration(duration)
                    .attr("x", 0)
                    .attr("y", 0)
                    .attr("width", 0)
                    .attr("height", 0);
            })
            .select('rect')
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 0)
            .attr("height", 0)
            .style("stroke-width", 0);

        var link = group_container.selectAll("path.link")
            .data(links, d => d.data.id);
        var linkEnter =  link.enter()
            .insert("path", "g")
            .attr("class", "link")
            .attr("d", () => {
                var o = {x: source.x0, y: source.y0};
                return diagonal(o, o);
            })
            .style("fill", "none")
            .style("stroke", "#ccc")
            .style("stroke-width", 3);
        var linkUpdate = linkEnter.merge(link);
        linkUpdate.transition()
            .duration(duration)
            .attr("d", d => diagonal(d, d.parent));
        var linkExit = link.exit()
            .transition()
            .duration(duration)
            .attr('d', () => {
                var o = {x: source.x, y: source.y};
                return diagonal(o, o);
            })
            .remove();

          nodes.forEach(function(d){
              d.x0 = d.x;
              d.y0 = d.y;
          });

        function diagonal(s, d) {
            return `M ${s.x} ${s.y}
                    C ${(s.x + d.x) / 2} ${s.y},
                      ${(s.x + d.x) / 2} ${d.y},
                      ${d.x} ${d.y}`;
        }
    }

    function delete_status() {
        var f = selected_status.parent;
        f.children.splice(f.children.indexOf(selected_status), 1);
        selected_status = null;
        that.draw();
    }

    function reload_status() {
        current_status = selected_status;
        LensView.load_status(current_status.id, current_status.node);
    }

    function collapse_status() {
        if (d.children) {
            d._children = d.children;
            d.children = null;
        } else {
            d.children = d._children;
            d._children = null;
        }
        update(d);
    }

    that._init = function(){
        // set up svg's width and height
        svg.attr("width", layout_width)
            .attr("height", layout_height)
            .on("click", () => {
                that.close_menu();
                if (that !== NavigationView) {
                    NavigationView.close_menu();
                } else if (another !== null) {
                    another.close_menu();
                }
            });
        group_container.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
        menu.style("cursor", "pointer")
            .style("visibility", "hidden");
        var menu_reload = menu.append("g")
            .attr("id", "nav-reload")
            .on("click", reload_status);
        menu_reload.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", menu_width)
            .attr("height", menu_height)
            .style("fill", "white")
            .style("stroke", "grey")
            .style("stroke-width", 2);
        menu_reload.append("text")
            .attr("x", menu_width / 2)
            .attr("y", menu_height / 2)
            .attr("width", menu_width)
            .attr("height", menu_height)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("stroke", "grey")
            .style("font-size", font_size)
            .text("Reload");
        var menu_delete = menu.append("g")
            .attr("id", "nav-delete")
            .attr("transform", "translate(" + 0 + ", " + menu_height + ")")
            .on("click", delete_status);
        menu_delete.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", menu_width)
            .attr("height", menu_height)
            .style("fill", "white")
            .style("stroke", "grey")
            .style("stroke-width", 2);
        menu_delete.append("text")
            .attr("x", menu_width / 2)
            .attr("y", menu_height / 2)
            .attr("width", menu_width)
            .attr("height", menu_height)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("stroke", "grey")
            .style("font-size", font_size)
            .text("Delete");
        // var menu_collapse = menu.append("g")
        //     .attr("id", "nav-collapse")
        //     .attr("transform", "translate(" + 0 + ", " + menu_height + ")")
        //     .on("click", collapse_status);
        // menu_collapse.append("rect")
        //     .attr("x", 0)
        //     .attr("y", 0)
        //     .attr("width", menu_width)
        //     .attr("height", menu_height)
        //     .style("fill", "white")
        //     .style("stroke", "grey")
        //     .style("stroke-width", 2);
        // menu_collapse.append("text")
        //     .attr("x", menu_width / 2)
        //     .attr("y", menu_height / 2)
        //     .attr("width", menu_width)
        //     .attr("height", menu_height)
        //     .attr("text-anchor", "middle")
        //     .attr("dominant-baseline", "middle")
        //     .style("stroke", "grey")
        //     .style("font-size", font_size)
        //     .text("Collapse");
    };

    that.init = function(){
        that._init();
    }.call();

    that.destroy = function() {
        svg.remove();
    };

    that.close_menu = function() {
        menu.style("visibility", "hidden");
        menu.selectAll("g").style("visibility", "hidden");
    };

    that.draw = function(){
        that.create();
        that.update();
        that.remove();
        if (another !== null) {
            another.draw();
        }
    };

    that.create = function(){
        var nav_data = null;
        var datatype = LensView.get_data_type();
        if (datatype === "train") {
            nav_data = train_nav;
        } else if (datatype === "test") {
            nav_data = test_nav;
        } else if (datatype === "all") {
            nav_data = all_nav;
        } else if (datatype === "both") {
            if (that === NavigationView) {
                nav_data = train_nav;
            } else {
                nav_data = test_nav;
            }
        }
        root = d3.hierarchy(nav_data.root, d => d.children);
        root.x0 = layout_width / 2;
        root.y0 = 0;
    };
    that.update = function(){
        treeData = treemap(root);
        update(root);
    };
    that.remove = function(){
    };

    that.load_data = function(id, data){
        if (id_set.has(id)) {
            return;
        } else {
            var new_status = {
                node: {
                    ...data,
                    encoding: compute_encoding(data.display_data),
                    nb_instances: data.display_data.length
                },
                id: id,
                parent: current_status,
                children: []
            };
            if (current_status === null) {
                var datatype = data.datatype;
                if (datatype === "train") {
                    train_nav.root = new_status;
                } else if (datatype === "test") {
                    test_nav.root = new_status;
                } else if (datatype === "all") {
                    all_nav.root = new_status;
                }
                current_status = new_status;
                id_set.add(id);
            } else {
                if (LensView.get_data_type() === "both" && data.datatype !== current_datatype) {
                    another.load_data(id, data);
                } else {
                    current_status.children.push(new_status);
                    current_status = new_status;
                    id_set.add(id);
                }
            }
        }
    };

    that.switch_datatype = function(_datatype) {
        if (_datatype === "train") {
            current_status = train_nav.root;
            current_datatype = "train";
        } else if (_datatype === "test") {
            current_status = test_nav.root;
            current_datatype = "test";
        } else if (_datatype === "all") {
            current_status = all_nav.root;
            current_datatype = "all";
        }
        return current_status;
    };

    that.resize = function() {
        svg.attr("height", layout_height / 2);
    };

    that.split = function() {
        another = new NavigationTree(container);
        another.inherit_data(train_nav, test_nav, all_nav, id_set);
        that.resize();
        another.resize();
        that.switch_datatype("train");
        another.switch_datatype("test");
    };
    that.merge = function() {
        another.destroy();
        another = null;
        svg.attr("height", layout_height);
    };
    that.inherit_data = function(_train_nav, _test_nav, _all_nav, _id_set) {
        train_nav = _train_nav;
        test_nav = _test_nav;
        all_nav = _all_nav;
        id_set = _id_set;
    };
    that.get_root = function(datatype) {
        if (datatype === "train") {
            return train_nav.root;
        } else if (datatype === "test") {
            return test_nav.root;
        } else {
            return all_nav.root;
        }
    };

    that.reset = function() {
        train_nav = {root: null};
        test_nav = {root: null};
        all_nav = {root: null};
        id_set = new Set([]);
        current_status = null;
        current_datatype = "test";
        selected_status = null;
        root = null;
    }
};