from flask import Flask, jsonify, send_file, request, render_template, send_from_directory
import numpy as np
import os
import sys
import json
from time import time

from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.ExchangePort import *

# get server root
SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)

# app = Flask(__name__, static_url_path="/static")
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def get_fav():
    filepath = os.path.join(SERVER_ROOT, "static")
    return send_from_directory(filepath, "favicon.ico")


@app.route("/api/manifest", methods=["GET"])
def app_get_manifest():
    # extract info from request
    from time import time
    t = time()
    dataname = request.args["dataset"]
    if dataname == "REA":
        return
    set_dataname(dataname)
    print("set_dataname time:", time() - t)
    manifest = get_manifest()
    logger.info("manifest, time cost: {}".format(time() - t))
    logger.debug("manifest: {}".format(manifest))
    return jsonify(manifest)


@app.route("/api/embed-data", methods=["GET"])
def app_get_data():
    embed_method = request.args["embed-method"]
    data = get_embed_data(embed_method)
    return jsonify(data)


@app.route("/api/idx", methods=["GET"])
def app_get_idx():
    data = get_idx()
    return jsonify(data)

@app.route("/api/feature", methods=["GET"])
def app_get_feature():
    data = get_feature()
    return jsonify(data)


@app.route("/api/label", methods=["GET"])
def app_get_label():
    data = get_label()
    return jsonify(data)

@app.route("/api/image", methods=["GET"])
def app_get_image():
    id = request.args["filename"].split(".")[0]
    id = int(id)
    if id < 0:
        return jsonify({
            "d": 1
        })
    image_path = get_image_path(id)
    return send_file(image_path)

@app.route("/api/thumbnail", methods=["GET"])
def app_get_thumbnail():
    id = request.args["filename"].split(".")[0]
    id = int(id)
    if id < 0:
        return jsonify({
            "d": 1
        })
    image_path = get_thumbnail_path(id)
    return send_file(image_path)

@app.route("/api/saliency-map", methods=["GET"])
def app_get_saliency_map():
    id = request.args["filename"].split(".")[0]
    id = int(id)
    if id < 0:
        return jsonify({
            "d": 1
        })
    saliency_map_path = get_saliency_map_path(id)
    return send_file(saliency_map_path)

@app.route("/api/original-samples", methods=["GET"])
def app_get_original_samples():
    data = get_original_samples()
    return jsonify(data)

@app.route("/api/grid-layout-query", methods=["GET"])
def app_grid_layout_query():
    embed_method = request.args["embed-method"]
    left_x = 0
    top_y = 0
    range_size = 1
    datatype = "train"
    try:
        left_x = float(request.args["left-x"])
        top_y = float(request.args['top-y'])
        range_size = float(request.args["range-size"])
        datatype = request.args["datatype"]
        logger.info("using info in url: right_x {}, top_y {}, range_size {}"
                    .format(left_x, top_y, range_size))
    except Exception as e:
        print("api info: ", e)
        logger.info("using default settings: right_x {}, top_y {}, range_size {}"
                    .format(left_x, top_y, range_size))
    b = get_grid_layout_query(embed_method, datatype, left_x, top_y, range_size)
    d = {
        "query": str(b)
    }
    return jsonify(d)

@app.route("/api/change-class", methods=["GET"])
def app_change_class():
    data = change_class()
    return jsonify(data)

@app.route("/api/grid-layout", methods=["GET"])
def app_get_grid_layout():
    embed_method = request.args["embed-method"]
    left_x = 0
    top_y = 0
    width = 1
    height = 1
    class_selection = None
    datatype = "train"
    distribution = ""
    node_id = -1
    try:
        left_x = float(request.args["left-x"])
        top_y = float(request.args['top-y'])
        width = float(request.args["width"])
        height = float(request.args["height"])
        datatype = request.args["datatype"]
        distribution = request.args["distribution"]
        node_id = int(request.args["node-id"])
        class_selection = request.args["class"]
        logger.info("using info in url: datatype: {}, right_x {}, "
                    "top_y {}, width {}, height {}, class_selection {}, node id: {}"
                    .format(datatype, left_x, top_y, width, height, class_selection, node_id))
    except Exception as e:
        print("api info: ", e)
        logger.info("using default settings: datatype: {}, right_x {}, "
                    "top_y {}, width {}, height {}, class_selection {}, node id: {}"
                    .format(datatype, left_x, top_y, width, height, class_selection, node_id))
    data = get_grid_layout_of_sampled_instances(embed_method, datatype, left_x,
                                                top_y, width, height, class_selection, node_id)
    data["distribution"] = distribution
    return jsonify(data)

@app.route("/api/decision-boundary", methods=["GET"])
def app_get_decision_boundary():
    # data_type = request.args["data-type"]
    # data = get_decision_boundary(data_type)
    data = get_decision_boundary_of_sampled_instances()
    return jsonify(data)

@app.route("/api/entropy/", methods=["GET"])
def app_get_entropy():
    data = get_entropy()
    return jsonify(data)

@app.route("/api/prediction/", methods=["GET"])
def app_get_prediction():
    data = get_prediction()
    return jsonify(data)

@app.route("/api/confidence/", methods=["GET"])
def app_get_confidence():
    data = get_confidence()
    return jsonify(data)

@app.route("/api/focus/", methods=["GET"])
def app_get_focus():
    id = int(request.args["id"])
    k = int(request.args["k"])
    data = get_focus(id=id, k=k)
    info = get_individual_info(id)
    data["info"] = info
    return jsonify(data)



def start_server(port=8183):
    app.run(port=port, host="0.0.0.0", threaded=True)


if __name__ == "__main__":
    start_server()
