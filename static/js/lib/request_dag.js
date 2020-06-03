/*
    default: binary type request
 */

var request_node = function (url, callback, type, method, headers, children, parents) {
    var myChildren = children || [];
    var myParents = parents || [];
    var myHeaders = headers || {
            "cache-control": "no-cache"
    };
    var myURL = url || "";
    var myMethod = method || "GET";
    var myType = type || "arraybuffer";
    var myStatus = "pending";
    // sleeping & pending & ready

    this.ready = function () {
        return myStatus === "ready";
    };

    this.pending = function() {
        return myStatus === "pending";
    };

    this.set_off = function() {
        myStatus = "sleeping";
    };

    this.set_on = function() {
        for (var i = 0; i < myParents.length; i ++) {
            if (!myParents[i].ready()) {
                myStatus = "pending";
                return;
            }
        }
        this.notify();
    };

    this.set_url = function(url) {
        myURL = url;
    };

    this.set_handler = function(handler) {
        callback = handler;
    };

    this._add_parent = function (p) {
        myParents.push(p);
    };

    this._add_child = function (c) {
        myChildren.push(c);
    };

    this.set_url = function (url) {
        myURL = url;
    };

    this.depend_on = function (n) {
        myParents.push(n);
        n._add_child(this);
        return this;
    };

    this.notify = function () {
        for (var i = 0; i < myParents.length; i ++) {
            if (!myParents[i].ready()) {
                return;
            }
        }
        var oReq = new XMLHttpRequest();
        console.log(myURL);
        oReq.open(myMethod, myURL, true);
        oReq.responseType = myType;
        for (var key in myHeaders) {
            if (myHeaders.hasOwnProperty(key)) {
                oReq.setRequestHeader(key, myHeaders[key]);
            }
        }

        oReq.onload = function (oEvent) {
            callback(oReq.response);
            myStatus = "ready";

            for (var i = 0; i < myChildren.length; i ++) {
                if (myChildren[i].pending())
                    myChildren[i].notify();
            }
        };
        oReq.send(null);
        return this;
    };
};
