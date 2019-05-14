var global_cfg = {
	query:null,
	results_json: null,
	// The url fragment that the solar server will response to; 
	urlcontext: "/apollo", 
	thechart: null, 
	max_ticks: 12,
	navheight:0,
	sources:null,
	current_source:0,
	date_picker_format: "mm/dd/yy",	
	datehandler: new DateHandler()
};

// Used to populate a select box. Allows user to specify a group-by clause in queries. 
var groupbyOptions =
	[
	    {"id": "0", "label": "None", "statisticsset": "2"},
        {"id": "yearmonthdayhourmin", "label": "By Date, Hour, and Minute (via database)",  "statisticsset":"1"},
        {"id": "yearmonthdayhour", "label": "By Date and Hour (via database)",  "statisticsset":"1"},
        {"id": "yearmonthday", "label": "By Year, Month, and Day (via database)",  "statisticsset":"1"},
        {"id": "yearmonth", "label": "By Year and Month (via database)",  "statisticsset":"1"},
		{"id": "monthofyear", "label": "By Month of Year (via database)",  "statisticsset":"1"},
		{"id": "dayofyear", "label": "By Day of Year (via database)",  "statisticsset":"1"},
		{"id": "dayhourofyear", "label": "By Day and Hour of Year (via database)", "statisticsset": "1" },
		{"id": "proc_yearmonthdayhourmin", "label": "By Date, Hour, and Minute  (via procedure)", "statisticsset":"0"},
		{"id": "proc_yearmonthdayhour", "label": "By Date and Hour  (via procedure)", "statisticsset":"0"},
		{"id": "proc_yearmonthday", "label": "By Year, Month, and Day (via procedure)", "statisticsset":"0"},
		{"id": "proc_yearmonth", "label": "By Year and Month (via procedure)", "statisticsset":"0"}
	];

/**
 * Generate query parameters based upon currently selected values, and invoke methods to 
 * query the remote server and create a chart or table from the response. 
 * @returns {undefined}
 */
function runQuery()
{
	var cfg = getConfig();
	
    if (!verifyDates())
    {
        return;
    }
    updateStatus("initiated...");
    try {

        var period;
        var periodText;
        var source;
        var srcpos;
        var thesite;
        var sitepos;
        var schema;
        var startParam;
        var stopParam;
        var statistic;
        var statisticLabels;
        var workingAttributeString;
        
		
		var sourceBox = document.getElementById('source_sel');
        source = sourceBox.value;
        
        schema = getUITemplate().name;

        var periodBox = document.getElementById('periodSelect');
        period = periodBox.value;
        periodText = periodBox[periodBox.selectedIndex].text;

        statistic = getSelectionList("statSelect", "statistic");
        statisticLabels = getSelectionValueString("statSelect");

        var siteselectbox = document.getElementById('siteSelect');
        thesite = siteselectbox.value;

        sitepos = siteselectbox.selectedIndex;
        
        workingAttributeString = getSelectionList("attributeSelect", "attribute");


        var startTemp = getStartDate();
        if (startTemp != null)
        {
			startParam = startTemp.getTime();
        } else
        {
            alert("Must select a valid start date!");
            return;
        }

        var stopTemp = getStopDate();
        if (stopTemp != null)
        {
            stopParam = stopTemp.getTime();
        } else
        {
            alert("Must select a valid stop date!");
            return;
        }

    cfg.datehandler.timegroupby = period;
  
	var timezoneBox = document.getElementById('timezoneSelect');
    cfg.datehandler.setTimezone(timezoneBox.value);
  var params =
            "groupby=" + period +
            "&source=" + source +
            "&site=" + thesite +
            "&schema=" + schema +
            "&start=" + startParam +
            "&stop=" + stopParam +
            workingAttributeString +
            statistic;

	cfg.query =
	{
            groupby:period ,
            source:source ,
            site:thesite ,
            schema:schema ,
            start:startParam ,
            stop:stopParam ,
            attributes:workingAttributeString ,
            statistics:statistic
	}
    updateResultsNote(periodText, statisticLabels);
    queryChart(params, cfg);
	
    } catch (err) {
        updateStatus(err.message);
    }
}
	

/**
 * Takes an object encoding parameters for a query to the server, 
 * and uses the return json results to create a chart or table. 
 * 
 * Query paremters have the following form: 
 * 
 * @param {type} params
 * @returns {undefined}
 */
function queryChart(params, cfg) {
    $.ajax({
        url: cfg.urlcontext + "?" + params,
        //force to handle it as text
        dataType: "text",
        success: function (data) {
				updateResults(cfg, data)
        },
        error: function (XMLHttpRequest, textStatus, errorThrown) {
            updateStatus(XMLHttpRequest.status + ": " + XMLHttpRequest.statusText);
        }
    });
}


function initSources(){
	var cfg = getConfig(); 
	var callback = function(data){
			cfg.sources = JSON.parse(data);
			var select = document.getElementById("source_sel");
			var i;
			for (i = select.options.length - 1; i >= 0; i--)
			{
				select.remove(i);
			}
			for (i = 0; i < cfg.sources.length; i++)
			{
				var option = document.createElement('option');
				option.value = cfg.sources[i].id;
				option.innerHTML = cfg.sources[i].label;
				select.appendChild(option);
			}
			select.selectedIndex = 0;
			initSourcesAux(0)
		}
	askServer("/sources","text", callback);
	document.getElementById("nav_div").addEventListener("mouseup", resizeChart);
	document.getElementById("results_container").addEventListener("mouseup", resizeChart);
}

function initSourcesAux(source_index){
	var cfg = getConfig(); 
	cfg.current_source = source_index;
	var callback = function(data){
		var temp = JSON.parse(data);
		if(!cfg.sources[source_index].schema_data){
			cfg.sources[source_index].schema_data = temp;
		}
		populateSources(); 
	}
	askServer("/source-ui?source="+cfg.sources[cfg.current_source].id+"&schema="+cfg.sources[cfg.current_source].schema,"text", callback);
}

function selectSource(){
	var select = document.getElementById("source_sel");
	var cfg = getConfig();
	var startdate = getStartDate();
	var stopdate = getStopDate();
	cfg.sources[cfg.current_source].initial_start = startdate.toISOString();
	cfg.sources[cfg.current_source].initial_stop = stopdate.toISOString();

	cfg.current_source = select.selectedIndex;

	var callback = function(data){
		cfg.sources[cfg.current_source].schema_data = JSON.parse(data);
		populateSources(); 
	}
	askServer("/source-ui?schema="+cfg.sources[cfg.current_source].schema,"text", callback);
}

function getUITemplate(){
	var cfg = getConfig(); 
	return cfg.sources[cfg.current_source].schema_data;
}

function askServer(query, dtype, callback){
	 $.ajax({
        url: query,
        //force to handle it as text
        dataType: "text",
        success: callback,
        error: function (XMLHttpRequest, textStatus, errorThrown) {
            updateStatus(XMLHttpRequest.status + ": " + XMLHttpRequest.statusText);
        }
    });
}

function updateResults(cfg, data){
			try{
				cfg.results_json = standardizeJSON(JSON.parse(data));
				populateShowHide(cfg.results_json);
				drawChart(cfg, 'results_div');
				updateStatus("");
			}
			catch (err) {
				updateStatus("error:" + err.message);
				var note = document.getElementById('results_note');
				note.innerHTML = "<i>Results could not be formatted as a chart. Showing as text instead.</i>";
				var results = document.getElementById('results_div');
				if(typeof data != "undefined"){
					results.appendChild(getResultsAsText(data));
				}
				else{
					results.innerText = "Results are undefined!";
				}
			}
}

function getResultsAsText(data){
	var ar = document.createElement('textarea');
	ar.id = "results_text_div";
    if(typeof data == "undefined"){
		ar.value = "Results are undefined!";
		
	}
	else if(typeof data == "string"){
		ar.value = "" + data
	}
	else{
		try{
			ar.value = "" + JSON.stringify(data,0) ;
		}
		catch(err){
			ar.value = "" + data
		}
	}
	return ar; 
}

/**
* Populate a select box using the column names of the results of a query. 
* This allows end-users to select which data series in the results to view. 
*/
function populateShowHide(resultsJSON){
	var columns = resultsJSON.columns;
	var show_div = document.getElementById("showHide");
	while (show_div.length > 0) {
		show_div.remove(show_div.length-1);
	}
	for (var i = 1; i < columns.length; i++)
    {
        var option = document.createElement('option');
        option.value = columns[i].label;
        option.innerHTML = columns[i].description;
		option.title = columns[i].description;
		option.selected = true;
        show_div.appendChild(option);
    }
}
	
/**
 * Query the server, determining whether it is online or not. 
 * The results change the text of the error message box in the web page. 
 */
function checkServer() {
    $.ajax({
        url: "/status",
        //force to handle it as text
        dataType: "text",
        success: function (data) {
            result = $.parseJSON(data);
			try {
					if(result.status == 1){
						updateStatus("server online");
					}
					else{
					updateStatus("server offline?");
					}
			}
			catch (err) {
                updateStatus("server offline?");
            }
        },
        error: function (XMLHttpRequest, textStatus, errorThrown) {
            updateStatus(XMLHttpRequest.status + ": " + XMLHttpRequest.statusText);
        }
    });
}


/**
Routine for drawing forecast data. Assumes model JSON is stored in model_data. 
*/
function drawForecastData()
{
	var cfg = getConfig(); 
    updateChartDivSize();
	//Redraw chart upon window resize; 
	document.getElementById("nav_div").addEventListener("mouseup", resizeChart);
	document.getElementById("results_container").addEventListener("mouseup", resizeChart);
	try{
		cfg.results_json = standardizeJSON(model_data);
		populateShowHide(cfg.results_json);
		drawChart(cfg, 'results_div');
		updateStatus("");
		}
	catch (err) {
		updateStatus("error:" + err.message);
		var note = document.getElementById('results_note');
		note.innerHTML = "<i>Results could not be formatted as a chart. Showing as text instead.</i>";
		var results = document.getElementById('results_div');
		if(model_data != "undefined"){
			results.appendChild(getResultsAsText(model_data));
		}
		else{
			results.innerText = "Results are undefined!";
			}
	}
}

/**
* A function to resize the displayed chart if the user resizes the navigation div. 
*/
function resizeChart(){
	var cfg = getConfig(); 
	if(cfg.thechart == null) {
		return;
		}
    var currentHeight = document.getElementById("nav_div").clientHeight;
	if(cfg.navheight == 0){
		cfg.navheight = currentHeight; 
	}
	if(cfg.navheight != currentHeight)
	{
		cfg.navheight = currentHeight; 
		drawChart(cfg, 'results_div');
	}
}
/**
 * Retrieve the div tag with id 'status_msg' and replace its current inner html
 * with the provided message. Used to provide status updates to the user. 
 * @param {type} msg The message to display to the user. 
 * @returns {undefined}
 */
function updateStatus(msg) {
    var status_ele = document.getElementById('status_msg');
    status_ele.innerHTML = msg;
}

/**
 * Retrieve the div tag with id 'results_note' and replace its current inner html
 * with the provided message. Used to provide status updates to the user. 
 * @param {type} msg The message to display to the user. 
 * @returns {undefined}
 */
function updateResultsNote(statInterval, statLabel) {
    var note = document.getElementById('results_note');
    if (statInterval == 0) {
        note.innerHTML = "";
    } else {
        note.innerHTML = "<b>Statistic</b>: " + statLabel + ". <b>Interval</b>: " + statInterval + ". Time points indicate the <i>beginning</i> time of the window used in calculating the statistic.";
    }
}


/**
 * Use the provided json data (encoding a table with selected metadata) to draw a chart or table. 
 * @param {type} jsonTableData
 * @returns {undefined}
 */
function drawChart(cfg, chartDivID) {
	
	//document.getElementById(chartDivID).innerHTML = "<pre>" + JSON.stringify(cfg.results_json,undefined, 2) + "</pre>"; return;
	
	var jsonTableData = cfg.results_json;
    
	if(jsonTableData == null){
		return;
	}
    var options;
    var chart;
    var site = jsonTableData.site;
	var startTime = jsonTableData.start;
    var endTime = jsonTableData.stop;
    var chartSelect = document.getElementById('chartSelect');
    var chartType = chartSelect.value;
	var dateflag = false;
	var thehaxis = null;
	var table = null;
	
    if(typeof site == "undefined" || site == null){
		site = ""
	}
	if(typeof startTime == "undefined" || startTime == null){
		startTime = ""
	}
	else {
		startTime = cfg.datehandler.formatTitleDateString(cfg.datehandler.createDate(startTime));
	}
	if(typeof endTime == "undefined" || endTime == null){
		endTime = ""
	}
	else{
		endTime = cfg.datehandler.formatTitleDateString(cfg.datehandler.createDate(endTime));	
	}
	var timeString = startTime;
	if(endTime != ""){
		timeString = timeString + " to " + endTime;
	}
	

	var chartTitle = site + " " + timeString;
	
    updateInnerHTML("results_title", chartTitle);

	var show_div = document.getElementById("showHide");

    if (jsonTableData.columns[0].type == "datetime" && chartType != "histogram")
    {
        dateflag = true;
        for (var i = 0; i < jsonTableData.rows.length; i++)
        {
            try
            {
                jsonTableData.rows[i][0] = cfg.datehandler.createDate(jsonTableData.rows[i][0]);
            } catch (err) {
                updateStatus(err.message);
            }
        }
    }
    if (dateflag) {
        thehaxis = getXTicksDateFormat(cfg.datehandler.getGroupBy());
    }
    options = {
        curveType: 'function',
        allowHtml: true,
        tooltip: {isHtml: true},
        explorer: {maxZoomIn: .01, maxZoomOut: 10, actions: ['dragToZoom', 'rightClickToReset']},
        legend: {position: 'bottom'},
        intervals: {'style': 'area'},
    };
    if (thehaxis != null) {
        options.hAxis = thehaxis;
    }

    table = new google.visualization.DataTable();

    if (chartType == "table")
    {

		table.addColumn(jsonTableData.columns[0]);
        for (var i = 1; i < jsonTableData.columns.length; i++){
			if(show_div.options[i-1].selected){
				table.addColumn(jsonTableData.columns[i]);
			}
        }
		var rowdata = [];
        for (var x = 0; x < jsonTableData.rows.length; x++)
        {
            var rowsubdata = [];
			rowsubdata.push(jsonTableData.rows[x][0]);
            for (var y = 1; y < jsonTableData.rows[x].length; y++)
            {
				if(show_div.options[y-1].selected){
					rowsubdata.push(jsonTableData.rows[x][y]);
				}
            }
            rowdata.push(rowsubdata);
        }
        table.addRows(rowdata);
		
        var formatter = getDateFormatter(cfg, document.getElementById('timezoneSelect').value);
		
		for (var x = 0; x < jsonTableData.rows.length; x++)
        {
			
			if(table.getValue(x, 0) instanceof Date){
				table.setFormattedValue(x,0,formatter.formatValue(table.getValue(x, 0)));
			}
        }

	}
	else if (chartType == "histogram"){
    
        for (i = 1; i < jsonTableData.columns.length; i++)
        {
			if(show_div.options[i-1].selected){
				table.addColumn(jsonTableData.columns[i]);
			}
            
        }
        var rowdata = [];
        for (var x = 0; x < jsonTableData.rows.length; x++)
        {
            var rowsubdata = [];
            for (var y = 1; y < jsonTableData.rows[x].length; y++)
            {
				if(show_div.options[y-1].selected){
					rowsubdata.push(jsonTableData.rows[x][y]);
				}
            }
            rowdata.push(rowsubdata);
        }
        table.addRows(rowdata);
    } 
	else {
        table.addColumn(jsonTableData.columns[0]);
        for (i = 1; i < jsonTableData.columns.length; i++)
        {
			if(show_div.options[i-1].selected){
				table.addColumn(jsonTableData.columns[i]);
				table.addColumn({type: 'string', role: 'tooltip', 'p': {'html': true}});
			}
        }
		var rowdata = [];
		try{
			for (var x = 0; x < jsonTableData.rows.length; x++){
				var rowsubdata = [];
				rowsubdata.push(jsonTableData.rows[x][0]);
				for (var y = 1; y < jsonTableData.rows[x].length; y++){
					if(show_div.options[y-1].selected){
						rowsubdata.push(jsonTableData.rows[x][y]);
						rowsubdata.push(formatCellValue(jsonTableData.columns[y], jsonTableData.rows[x][0], jsonTableData.rows[x][y], cfg.datehandler));
					}
				}
				rowdata.push(rowsubdata);
			}
		}	
		catch(err){
			alert(err); 
		}
		table.addRows(rowdata);
    }

    if (chartSelect != null)
    {
        if (chartType == "column")
        {
            chart = new google.visualization.ColumnChart(document.getElementById(chartDivID));
            options.bar = {groupWidth: '100%'};
        } else if (chartType == "area")
        {
            chart = new google.visualization.AreaChart(document.getElementById(chartDivID));
        } else if (chartType == "scatter")
        {
            chart = new google.visualization.ScatterChart(document.getElementById(chartDivID));
        } else if (chartType == "stepped")
        {
            chart = new google.visualization.SteppedAreaChart(document.getElementById(chartDivID));
        } else if (chartType == "histogram")
        {
            chart = new google.visualization.Histogram(document.getElementById(chartDivID));
        } else if (chartType == "table")
        {
            chart = new google.visualization.Table(document.getElementById(chartDivID));
            options = {
                showRowNumber: true,
            };
        } else
        {
            chart = new google.visualization.LineChart(document.getElementById(chartDivID));
        }
    } else
    {
        chart = new google.visualization.LineChart(document.getElementById(chartDivID));
    }
	
	if(chartType != "histogram" && cfg.datehandler.useUTC){
		xTicks = getXTicks(table,  cfg.max_ticks, cfg.datehandler);
		options.hAxis = {ticks: xTicks};
	}

    updateChartDivSize();

    //add link to png image of chart. 
    google.visualization.events.addListener(chart, 'ready', function () {
        var outfile_span = document.getElementById('outfile_span');
        if (outfile_span != null && chartType != "table") {
			var png_text =  '<a id="imglink" href="' + chart.getImageURI() + '"  target="_blank">png</a>';
        } else {
            var png_text = ""; 
        }
		//csvlink = "<span  onclick='openAsCSV()' style='color:blue;text-decoration:underline;cursor:pointer;'>(open csv)</span>";
		csvlink2 = "<span  onclick='saveAsCSV()' style='color:blue;text-decoration:underline;cursor:pointer;'>csv</span>";
		jsonlink = "<span  onclick='saveAsJSON()' style='color:blue;text-decoration:underline;cursor:pointer;'>json</span>";
		cb = "<span  onclick='saveToClipboard()' style='color:blue;text-decoration:underline;cursor:pointer;'>clipboard</span>";
		outfile_span.innerHTML = png_text + " " + csvlink2 + " " + jsonlink + " " + cb;
    });

    chart.draw(table, options);
    cfg.thechart = chart;
}


function standardizeJSON(jsonIn, hasHeaders){
	var result = {}; 
	if(jsonIn instanceof Array){
		result.site = null; 
		result.start = null; 
		result.stop = null; 
		result.rows = jsonIn;
		result.columns = null; 
	}
	else{
		result = jsonIn; 
	}
	if(result.columns == null && result.rows.length > 0){
		result.columns = []; 
		var headers = null;
		var startRow = 0;
		if(hasHeaders){
			headers = result.rows.shift();
			startRow = 1; 
		}
		for(var i = 0; i < result.rows[startRow].length;i++){
			var col = {};
			if(hasHeaders){
				col.label = "" + headers[i];
			}
			else{
				col.label = "series " + i; 				
			}
			col.type = getChartDataType(result.rows[startRow][i]);
			result.columns.push(col);
		}
	}
	return result;
}

function getChartDataType(val){
	if(typeof val == "number"){ return "number";}
	if(typeof val == "boolean"){ return "boolean";}
	if(typeof val == "date"){ return "date";}
	if(typeof val == "datetime"){ return "datetime";}
	if(typeof val == "timeofday"){ return "timeofday";}
	if(typeof val == "string"){
		var res = Date.parse(val);
		if (!isNaN(res)) return "datetime";
		}
	return "string";
	}

/**
* A method to manually calculate x-axis ticks for a chart. 
* This is needed to counteract Google chart's rendering dates in the ticks using the local time zone.  
*/
function getXTicks(table, max_ticks, datehandler){
		var xTicks = [];
		var rowNumbers = table.getNumberOfRows();
		var m;
		if(rowNumbers < max_ticks * 2 || rowNumbers < 20){
			m = 1;
		}
		else{
			m = Math.floor(rowNumbers/max_ticks);
		}

		var last = 0; 
		for (var i = 0; i < rowNumbers; i++) {
			if( i % m == 0)
			{
			xTicks.push({
					v: table.getValue(i, 0),
					f: datehandler.formatDatetoLocaleDateString(table.getValue(i, 0))
					});
			}
		}
	return 	xTicks;
}

/**
* Redraw the query results. 
*/
function refreshChart(){
	var cfg = getConfig(); 
	if(cfg.results_json != null){
		cfg.datehandler.setTimezone(document.getElementById('timezoneSelect').value);
		drawChart(cfg, 'results_div');
        updateStatus("");
	}
}
     
/**
* Governs how dates are displayed in the x-axis of google charts. 
*/
function getXTicksDateFormat(timegroupby)
{
    var thehaxis = {format: 'M/d/yy-HH:mm ZZZZ', gridlines: {count: -1, units: {days: {format: ['MMM dd ZZZZ']}, hours: {format: ['HH:mm', 'ha ZZZZ']}}}};
	if(timegroupby == null) return thehaxis; 

	switch(timegroupby){
		case "yearmonthdayhourmin":
			return {format: 'M/d/yy-HH:mm ZZZZ'};
		case "yearmonthdayhour":
			return {format: 'M/d/yy-HH ZZZZ'};
		case "yearmonthday":
			return {format: 'M/d/yy'};
		case "yearmonth":
			return {format: 'M/yy'};
		case "dayhourofyear":
			return {
				format: 'MMM dd HH:mm',
					gridlines: {count: -1, units: {
                    years: {format: ['MMM']},
                    months: {format: ['MMM']},
                    days: {format: ['MMM dd']},
                    hours: {format: ['MMM dd HH:mm']},
                    minutes: {format: ['MMM dd HH:mm']}
                }
            }};
		case "dayofyear":
		 return {
            format: 'MMM dd',
            gridlines: {count: -1, units: {
                    years: {format: ['MMM']},
                    months: {format: ['MMM']},
                    days: {format: ['MMM dd']},
                    hours: {format: ['MMM dd']},
                    minutes: {format: ['MMM dd']}
                }
            }};
		case "monthofyear":
		return {
            format: 'MMM',
            gridlines: {count: -1, units: {
                    years: {format: ['MMM']},
                    months: {format: ['MMM']},
                    days: {format: ['MMM']},
                    hours: {format: ['MMM']},
                    minutes: {format: ['MMM']}
                }
            }};
		
	}
	return thehaxis; 
}

/**
* Given various grouping options, returns a formatter for formatting dates.
* This controls which elements of a date are displayed. For instance, if 
* the data records average irradiance for each month, only the month should be displayed. 
*
* This is really only used for table charts. 
*/
function getDateFormatter(cfg, displayFormat)
{
    
	var tz = cfg.datehandler.getTimeZoneOffset();

	if(displayFormat === "unix"){
		return {formatValue:function(e){
			if(e instanceof Date){
				return Math.floor(e.getTime()/1000)
			}
			else{
				return e;
			}
			}};
		}
	else if(displayFormat === "local"){
		return {formatValue:function(e){
			if(e instanceof Date){
				return cfg.datehandler.formatDatetoLocaleDateString(e);
			}
			else{
				return e;
			}
			}};
		}
	switch(cfg.datehandler.timegroupby){
	case "dayofyear":
		return new google.visualization.DateFormat({pattern: 'MM-dd', timeZone: tz});
	case "dayhourofyear":
		return new google.visualization.DateFormat({pattern: 'MM-ddTHH:mm:ssZZZ', timeZone: tz});
    case "monthofyear":
		return new google.visualization.DateFormat({pattern: 'MM', timeZone: tz});
    case "yearmonthdayhourmin":
		return new google.visualization.DateFormat({pattern: 'yyyy-MM-ddTHH:mm:ssZZZ', timeZone: tz});
	case "yearmonthdayhour":
		return new google.visualization.DateFormat({pattern: 'yyyy-MM-ddTHH:mm:ssZZZ', timeZone: tz});
	case "yearmonthday":
		return new google.visualization.DateFormat({pattern: 'yyyy-MM-dd', timeZone: tz});
	case "yearmonth":
		return new google.visualization.DateFormat({pattern: 'yyyy-MM', timeZone: tz});
	default:
		return new google.visualization.DateFormat({pattern: 'yyyy-MM-ddTHH:mm:ssZZZ', timeZone: tz});
	}
}

/**
 * Generates a formatted version of a data value using the provided date and 
 * metadata for the value. This is used in creating tooltips for points in charts. 
 * @param {type} metadata An object storing the values label, longname (description), and units 
 * @param {type} thedate A long value representing a time point. 
 * @param {type} value The value to format. 
 * @returns {String}
 */
function formatCellValue(metadata, thedate, value,datehandler)
{
	
	var s = "<div class='tooltip_date'>date: "
				+  datehandler.formatDatetoLocaleDateString(thedate)
                + '</div>' 
				+ "<div class='tooltip_value'>value: "
                + value
                + '</div>';
	for (const [ key, value ] of Object.entries(metadata)) {
		s = s + "<div class='tooltip_value'>" + key + ": " + value + "</div>";
	}
	return s; 
		/*return "<div class='tooltip_date'>Date: "
                //+  datehandler.formatDatetoLocaleDateString(datehandler.createDate(thedate))
				+  datehandler.formatDatetoLocaleDateString(thedate)
                + '</div>'
                + "<div class='tooltip_label'>Name: "
                + metadata.label
                + '</div>'
                + "<div class='tooltip_longname'>Description: "
                + metadata.description
                + '</div>'
                + "<div class='tooltip_value'>Value: "
                + value
                + '</div>'
                + "<div class='tooltip_units'>"
                + "Base Units: "
                + metadata.units
                + '</div>';
				*/
}

/**
 * Returns a string representation of a list of option IDs selected in the specified select input with ID eleID. 
 *
 * Result has the form key=Val1&key=Val2&...&key=ValN, where key is passed as an input argument. 
 */
function getSelectionList(eleID, key)
{
    var workingAttributeString;
    var selectbox = document.getElementById(eleID);
    if (selectbox != null)
    {
        workingAttributeString = "";
        for (var i = 0; i < selectbox.options.length; i++)
        {
            if (selectbox[i].selected)
            {
                workingAttributeString = workingAttributeString + "&" + key + "=" + selectbox[i].value;
            }
        }
    } else
    {
        workingAttributeString = key + "=" + selectbox[0].value;
    }
    return workingAttributeString;
}
/**
 * Returns a string representation of a list of option IDs selected in the specified select input with ID eleID. 
 * Result has the form Val1,Val2,...ValN.
 */
function getSelectionValueString(eleID)
{
    var workingAttributeString;
    var selectbox = document.getElementById(eleID);
    if (selectbox != null)
    {
        workingAttributeString = "";
        for (var i = 0; i < selectbox.options.length; i++)
        {
            if (selectbox[i].selected) {
                if (workingAttributeString == "") {
                    workingAttributeString = selectbox[i].value;
                } else {
                    workingAttributeString = workingAttributeString + "," + selectbox[i].value;
                }
            }
        }
    } else
    {
        workingAttributeString = selectbox[0].value;
    }
    return workingAttributeString;
}

/**
 * Populates the select box widgets with information on database sources, their 
 * associated sites and modules, and their associated attributes. This should be invoked once
 * when the page is first loaded. 
 * @returns {undefined}
 */
function populateSources()
{
    var sourceDiv = document.getElementById('sources');
	
	while (sourceDiv.firstChild) {
		sourceDiv.removeChild(sourceDiv.firstChild);
	}
	
	var source = createSource(getUITemplate().name);
	
	sourceDiv.appendChild(source);
	source.style.display = 'block';
    populateSites();
	populatePeriods();
	setInitialDates();
    updateDateFormat();
	updateStatisticsOptions();
    toggleStatisticSelectEnabled();
    updateChartDivSize();
	checkServer();
}

/**
 * Update the size of the displayed chart based upon the current size of the 'nav_div' element. 
 * Called (typically when a page is resized). 
 * @returns {undefined}
 */
function updateChartDivSize()
{
    document.getElementById('results_div').style.height = ($('#nav_div').height() - 60) + 'px';
}

/**
 * For a given string srcName, create a set of widgets associated with the data source named by srcName. 
 * The created widgets (mostly select boxes) allow the user to specify parameters for queries to the source. 
 
 * @param {type} srcName
 * @returns {Element|createSource.source}
 */
function createSource(srcName)
{
    var source = document.createElement('div');
    source.id = srcName;
    source.style.display = 'none';

    var siteHeader = document.createElement('div');
    siteHeader.innerHTML = "Module/Site";
    siteHeader.className = "param_header";

    var paramHeader = document.createElement('div');
    paramHeader.innerHTML = "Parameter";
    paramHeader.className = "param_header";

    var periodHeader = document.createElement('div');
    periodHeader.innerHTML = "Group By";
    periodHeader.className = "param_header";

    var statHeader = document.createElement('div');
    statHeader.innerHTML = "Statistic";
    statHeader.className = "param_header";

    var siteSelect = document.createElement('select');
    siteSelect.id = "siteSelect";
    siteSelect.className = "select_widget";
    siteSelect.addEventListener("change", populateSiteAttributeOptions);

    var attributeSelect = document.createElement('select');
    attributeSelect.id = "attributeSelect";
    attributeSelect.className = "select_widget";
    attributeSelect.multiple = true;

    var periodSelect = document.createElement('select');
    periodSelect.id = "periodSelect";
    periodSelect.className = "select_widget";
    periodSelect.addEventListener("change", updateDateFormat);
    
    var statSelect = document.createElement('select');
    statSelect.id = "statSelect";
    statSelect.className = "select_widget";
    statSelect.multiple = true;
	
    source.appendChild(siteHeader);
    source.appendChild(siteSelect);

    source.appendChild(paramHeader);
    source.appendChild(attributeSelect);

    source.appendChild(periodHeader);
    source.appendChild(periodSelect);

    source.appendChild(statHeader);
    source.appendChild(statSelect);
    return source;
}

/**
* Based upon whether the user has selected to group query results, populate the statistics select box with 
* statistics options compatible with the grouping. 
*/
function updateStatisticsOptions()
{
    var statSelect = document.getElementById('statSelect');
    var periodSelect = document.getElementById('periodSelect');
	
	var statset = 2;
	var i;
	var groupbyOptions = getGroupByOptions();
	for(i = 0; i < groupbyOptions.length;i++)
	{
		if(periodSelect.selectedIndex == i)
		{
			statset = groupbyOptions[i].statisticsset; 
			break; 
		}
	}
var stats = [["None", "None"]];

if(statset == 0)
{
     stats = [
        ["MEAN", "Mean"],
        ["MIN", "Minimum"],
        ["MAX",  "Maximum"],
        ["SUM", "Sum"],
		["COUNT", "Count"],
        ["PER5", "5th Percentile"],
        ["PER10","10th Percentile"],
        ["PER25","25th Percentile"],
        ["PER50","50th Percentile"],
        ["PER75","75th Percentile"],
        ["PER90","90th Percentile"],
        ["PER95","95th Percentile"],
        ["PER99","99th Percentile"],
        ["STDP", "Standard Deviation (Population)"],
        ["STD", "Standard Deviation (Sample)"],
        ["VARP", "Variance (Population)"],
        ["VAR",  "Variance (Sample)"]
    ];
}
else if(statset == 1){
	stats = [
			["AVG", "Mean"],
			["MIN","Minimum"],
			["MAX", "Maximum"],
			["SUM",  "Sum"],
			["COUNT",    "Count"]
		];
	}
 
    for (i = statSelect.options.length - 1; i >= 0; i--)
    {
        statSelect.remove(i);
    }
    for (i = 0; i < stats.length; i++)
    {
        var option = document.createElement('option');
        option.value = stats[i][0];
        option.innerHTML = stats[i][1];
        statSelect.appendChild(option);
    }
    statSelect.selectedIndex = 0;
}


/**
 * Enable or disable the statistics select box based upon whether a nonzer statistics interval is selected. 
 * If the interval is 0, then the box is disabled, otherwise it's enabled. 
 * @returns {undefined}
 */
function toggleStatisticSelectEnabled()
{
    var selectBox = document.getElementById('periodSelect');
    var statBox = document.getElementById('statSelect');
    if (selectBox.selectedIndex == 0)
    {
        statBox.disabled = true;
    } else {
        statBox.disabled = false;
    }
}


/**
 * Populates the site select box. 
 * Indirectly, this populates the attribute select box (which is tied to the site). 
 * @returns {undefined}
 */
function populateSites()
{
    var siteStr = 'siteSelect';
    var select = document.getElementById(siteStr);
    var modulesArray = getUITemplate().tables;
    var i;
    for (i = select.options.length - 1; i >= 0; i--)
    {
        select.remove(i);
    }
    for (i = 0; i < modulesArray.length; i++)
    {
        var option = document.createElement('option');
        option.value = modulesArray[i].id;
        option.innerHTML = modulesArray[i].label;
        select.appendChild(option);
    }
    select.selectedIndex = 0;
    populateSiteAttributeOptionsAux(0);
}




/**
 * When called, finds the selected source and site and then invokes populateSiteAttributeOptionsAux to populate the appropriate
 * attribute select box. 
 * @returns {undefined}
 */
function populateSiteAttributeOptions()
{
    var siteSelectBox = document.getElementById('siteSelect');
    populateSiteAttributeOptionsAux(siteSelectBox.selectedIndex);
}


/**
 * Takes the given source index, name, and site index and populate the 
 * attribute select box assocated with the source. The site is used to obtain the correct attribute set. 
 
 * @param {type} srcIndex
 * @param {type} srcName
 * @param {type} siteIndex
 * @returns {undefined}
 */
function populateSiteAttributeOptionsAux(siteIndex)
{
    var attributeSelectBox = document.getElementById('attributeSelect');
    var attributeSetArray = getUITemplate().columns;
    var attributes;
	
    if (attributeSetArray.length > 1)
    {
        attributes = attributeSetArray[siteIndex];
    } else
    {
        attributes = attributeSetArray[0];
    }

    var i;
    for (i = attributeSelectBox.options.length - 1; i >= 0; i--)
    {
        attributeSelectBox.remove(i);
    }
    for (i = 0; i < attributes.length; i++)
    {
        var option = document.createElement('option');
        option.value = attributes[i].id;
        option.innerHTML = attributes[i].label;
		option.title = attributes[i].label;
        attributeSelectBox.appendChild(option);
    }
    attributeSelectBox.selectedIndex = 0;
}

/**
 * Using the given source index and stored template data (a javascript array), populate the 
 * associated period select box with values for periods associated with the source. 
 * each source is stored in the array and possesses a 'period' property (an array of strings enumerating permissible periods). 
 * @param {type} srcPos
 * @returns {undefined}
 */
function populatePeriods()
{
    var periodSelectBox = 'periodSelect';
    var select = document.getElementById(periodSelectBox);
    var groupbyOptions = getGroupByOptions();
    var i;
	
    for (i = select.options.length - 1; i >= 0; i--)
    {
        select.remove(i);
    }
    for (i = 0; i < groupbyOptions.length; i++)
    {
        var option = document.createElement('option');
        option.value = groupbyOptions[i].id;
        option.innerHTML = groupbyOptions[i].label;
        select.appendChild(option);
    }
    select.selectedIndex = 0;
}


/**
* Ensures start time comes before stop time; other constraints could be added here to ensure queries don't take too long. 
*/
function verifyDates()
{
    try {
        var periodSelectBox = document.getElementById('periodSelect');
        var maximumGapForDateHourMinuteGrouping = 14;
        var maximumGapForDateHourGrouping = 60;
        var maximumGapForDateGrouping = 365;
        var maximumGapForYearMonthGrouping = 2 * 365;
        var oneday = 1000 * 60 * 60 * 24;
        var grouping = periodSelectBox.options[periodSelectBox.selectedIndex].value;

        var startTime = getStartDate(); 
        var stopTime = getStopDate(); 

        if (startTime > stopTime)
        {
            alert("Invalid dates! Please ensure that the start date is before the stop date.");
            return false;
        }/*
        if ("yearmonthdayhourmin" == grouping && (stopTime - startTime) > (oneday * maximumGapForDateHourMinuteGrouping))
        {
            alert("Invalid dates! For this grouping, please ensure that the start and stop dates are no more than " + maximumGapForDateHourMinuteGrouping + " days apart.");
            return false;
        }
        if ("yearmonthdayhour" == grouping && (stopTime - startTime) > (oneday * maximumGapForDateHourGrouping))
        {
            alert("Invalid dates! For this grouping, please ensure that the start and stop dates are no more than  " + maximumGapForDateHourGrouping + "  days apart.");
            return false;
        }
        if ("yearmonthday" == grouping && (stopTime - startTime) > (oneday * maximumGapForDateGrouping))
        {
            alert("Invalid dates! For this grouping, please ensure that the start and stop dates are no more than  " + maximumGapForDateGrouping + "  days apart.");
            return false;
        }
        if ("yearmonth" == grouping && (stopTime - startTime) > (oneday * maximumGapForYearMonthGrouping))
        {
            alert("Invalid dates! For this grouping, please ensure that the start and stop dates are no more than  " + maximumGapForYearMonthGrouping + "  days apart.");
            return false;
        }*/
    } catch (err)
    {
        updateStatus(err);
    }
    return true;
}



/**
 * replace the inner HTML of the element with ID elementID with val. 
 * @param {type} elementID
 * @param {type} val
 * @returns {undefined}
 */
function updateInnerHTML(elementID, val)
{
    try {
        var element = document.getElementById(elementID);
        element.innerHTML = val;
    } catch (err) {
    }
}




function getGroupByOptions()
{
    return groupbyOptions;
}

/**
 * Set intial dates for the query. User can override them. 
 * @returns {undefined}
 */
function setInitialDates()
{
	
	var sourceSelect = document.getElementById('source_sel');
	var cfg = getConfig();
	var startd;
    var stopd;
	if(cfg.sources){
		source = cfg.sources[sourceSelect.selectedIndex];
		if(source.initial_start){
			startd = cfg.datehandler.createDate(source.initial_start);
		}
		if(source.initial_stop){
			stopd = cfg.datehandler.createDate(source.initial_stop);
		}
	}
	if(!startd){
		startd = cfg.datehandler.createDate(2017,0,1);
	}
	if(!stopd){
		startdLong = startd.getTime();
		var nextLong = startdLong + (24 * 60 * 60 * 1000);
		stopd = cfg.datehandler.createDate(nextLong);
	}
	setStartDate(startd);
	setStopDate(stopd);  
}

/**
 * Gets the start date chosen from the date picker. 
 */
function getStartDate()
{
var cfg = getConfig();
var thedate = getValueAsDate('startdate').getTime();
var hourbox = document.getElementById('starthour');
return cfg.datehandler.createDate(thedate + 1000*60*60*hourbox.options[hourbox.selectedIndex].value);
}

/**
 * Gets the stop date chosen from the date picker. 
 */
function getStopDate()
{
var cfg = getConfig();
var thedate = getValueAsDate('stopdate').getTime();
var hourbox = document.getElementById('stophour');
return cfg.datehandler.createDate(thedate + 1000*60*60*hourbox.options[hourbox.selectedIndex].value);
}

/**
 * Sets the start date for the date picker. 
 */
function setStartDate(thedate)
{
	var cfg = getConfig();
    var startwidget = document.getElementById('startdate');
    startwidget.value =  cfg.datehandler.formatDate(cfg.date_picker_format, thedate);
}

/**
 * Sets the stop date for the date picker. 
 */
function setStopDate(thedate)
{
	var cfg = getConfig();
    var startwidget = document.getElementById('stopdate');
    startwidget.value =  cfg.datehandler.formatDate(cfg.date_picker_format, thedate);
}

/**
* Reads a formatted date string from the given element and parses it as a date. 
*/
function getValueAsDate(elementID)
{
	var cfg = getConfig();
    var picker = document.getElementById(elementID);
    var parts = picker.value.split("/");
    var year = 2016 // (use a leap year); 
    var month = parts[0] - 1;
    var day = parts[1];
    // if year is provided in input, use it instead of 2016. 
    if (parts.length == 3)
    {
        year = parts[2];
    }
    var thedate = cfg.datehandler.createDate(year, month, day);
    return thedate;
}

function updateDateFormat()
{
  var theformat = getConfig().date_picker_format; 
  $("#startdate").datepicker("option", "dateFormat", theformat);
  $("#stopdate").datepicker("option", "dateFormat", theformat);
  updateStatisticsOptions();
  toggleStatisticSelectEnabled();
}

function openAsCSV() {
	var cfg = getConfig(); 
	if(cfg.results_json == null)
		return; 
	var jsonTableData = cfg.results_json;
	var w = window.open();
	w.document.open();
	w.document.write(writeCSVString(jsonTableData,",","<br/>", cfg));
	w.document.close();
}


function saveAsCSV() {
	var cfg = getConfig(); 
	if(cfg.results_json == null)
		return; 
	var jsonTableData = cfg.results_json;
	var str = writeCSVString(jsonTableData,",","\n", cfg);
	var element = document.createElement('a');
	element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(str));
	element.setAttribute('download', get_output_filename(cfg.query,'.csv'));
	element.style.display = 'none';
	document.body.appendChild(element);
	element.click();
	document.body.removeChild(element);
	updateStatus("csv saved to file.");
}

function saveAsJSON() {
	var cfg = getConfig(); 
	if(cfg.results_json == null)
		return; 
	var jsonTableData = cfg.results_json;
	var oldrows = jsonTableData.rows;
	var newrows = []; 
	var newrows = Array.from({length: oldrows.length});
    for(var i = 0; i < oldrows.length;i++){
    	var newrow = Array.from({length: oldrows[i].length});
    	for(var j = 0; j < oldrows[i].length;j++){
    		if(oldrows[i][j] instanceof Date){
        		var res = cfg.datehandler.formatDatetoLocaleDateString(oldrows[i][j]);
        		var n = Number(res); 
        		if(isNaN(n)){
            		newrow[j] = res;
            		}
        		else {
            		newrow[j]= n; 
                }
            }
            else{
            newrow[j] = oldrows[i][j];
            }
        }
        newrows[i] = newrow; 
    }
    jsonTableData.rows = newrows; 

	var element = document.createElement('a');
	element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(JSON.stringify(jsonTableData)));
	jsonTableData.rows = oldrows; 
	element.setAttribute('download', get_output_filename(cfg.query,'.json'));
	element.style.display = 'none';
	document.body.appendChild(element);
	element.click();
	document.body.removeChild(element);
	updateStatus("json saved to file.");
}

function saveToClipboard() {
	var cfg = getConfig(); 
	if(cfg.results_json == null)
		return; 
	var jsonTableData = cfg.results_json;
	var oldrows = jsonTableData.rows;
	var newrows = []; 
	var newrows = Array.from({length: oldrows.length});
    for(var i = 0; i < oldrows.length;i++){
    	var newrow = Array.from({length: oldrows[i].length});
    	for(var j = 0; j < oldrows[i].length;j++){
    		if(oldrows[i][j] instanceof Date){
        		var res = cfg.datehandler.formatDatetoLocaleDateString(oldrows[i][j]);
        		var n = Number(res); 
        		if(isNaN(n)){
            		newrow[j] = res;
            		}
        		else {
            		newrow[j]= n; 
                }
            }
            else{
            newrow[j] = oldrows[i][j];
            }
        }
        newrows[i] = newrow; 
    }
    jsonTableData.rows = newrows; 
	var ele = document.createElement('textarea');
	ele.textContent = JSON.stringify(jsonTableData);

    jsonTableData.rows = oldrows; 

	document.body.appendChild(ele);
	
	var selection = document.getSelection();
	var range = document.createRange();
	range.selectNode(ele);
	selection.removeAllRanges();
	selection.addRange(range);
	document.execCommand('copy')
	selection.removeAllRanges();
	document.body.removeChild(ele);
	updateStatus("json copied to clipboard.");
}

function writeCSVString(jsonTableData, delimeter, newline_char,cfg) {
	var s = ""; 
	if (delimeter === undefined){
		delimeter = ","
	}
	if (newline_char === undefined){
		newline_char = "\n"
	}
	if(jsonTableData == null)
		return; 
	for (var c = 0; c < jsonTableData.columns.length; c++){
		s = s + jsonTableData.columns[c].label + delimeter
	}
	s = s + newline_char;
	for (var x = 0; x < jsonTableData.rows.length; x++){
		for(var y = 0; y < jsonTableData.rows[x].length;y++){
			if(jsonTableData.rows[x][y] instanceof Date){
				s = s + cfg.datehandler.formatDatetoLocaleDateString(jsonTableData.rows[x][y])+ delimeter;
			}
			else s = s + jsonTableData.rows[x][y]+ delimeter;
		}
		s = s + newline_char;
	}
	return s; 
}

function get_output_filename(query,extension)
{
   if(query == null) 
	   return "data"+extension;
	var cfg = getConfig();
	var startStr = cfg.datehandler.createDate(query.start).toISOString().replace(/:/g, '_');
	var stopStr =  cfg.datehandler.createDate(query.stop).toISOString().replace(/:/g, '_');
	return  query.source  + " " + startStr + " to "+ stopStr + extension;
}

function getConfig(){
	return global_cfg; 
}

//Redraw chart upon window resize; 
$(window).resize(function () {
	var cfg = getConfig();
	if(cfg.thechart != null){
		drawChart(getConfig(), 'results_div');
	}
});

$(function () {
    $("#startdate").datepicker({
        changeYear: false,
        dateFormat: getConfig().date_picker_format,
    })

});

$(function () {
    $("#stopdate").datepicker({
        changeYear: false,
        dateFormat: getConfig().date_picker_format,
    })
});



