

class DateHandler {

    constructor(timezone) {

        this.useUTC = false;

        this.titleDateFormat = "local";

        this.monthFormat = "long";

        this.timegroupby = "none";

        this.locale = undefined;

        this.setTimezone(timezone);

		this.month_strings = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

		this.month_strings_short = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    }



    setTimezone(timezone) {

        if ("utc" == timezone) {

            this.useUTC = true;

            this.titleDateFormat = 'utc';

        }

        else if ("local" == timezone) {

            this.useUTC = false;

            this.titleDateFormat = 'local';

        }

        else if ("unix" == timezone) {

            this.useUTC = true;

            this.titleDateFormat = 'unix';

        }

    }

    /**

	* Given a date object, set its hours, minutes, and seconds values to 0. 

	* @param {type} d

	* @returns {unresolved}

	*/

    trimDate(d) {



        if (this.useUTC) {

            d.setUTCDate(d.getUTCDate());

            d.setUTCHours(0);

            d.setUTCMinutes(0);

            d.setUTCSeconds(0);

            return d;

        }

        d.setDate(d.getDate());

        d.setHours(0);

        d.setMinutes(0);

        d.setSeconds(0);

        return d;

    }





    createDate(a, b, c) {

        if (typeof a != 'undefined' && typeof b != 'undefined' && typeof c != 'undefined') {

            // order: year, month, day

            // month: 0-11, day 1-31

            // Need to use Date.UTC to specify creation of UTC time rather than local. 

            if (this.useUTC) {

                return new Date(Date.UTC(a, b, c));

            }

            else {

                return new Date(a, b, c);

            }

        }

        if (typeof a != 'undefined') {

            if (typeof a == 'string') {

                // strings are parsed as UTC by default?

                return new Date(a);

            }

            //otherwise milliseconds since Jan 1 1970 UTC is assumed. 

            return new Date(a);

        }

        // return now

        return new Date();

    }



    formatDate(theformat, thedate) {

        var theyear;

        var themonth;

        var theday;

        if (this.useUTC) {

            theyear = thedate.getUTCFullYear();

            themonth = thedate.getUTCMonth() + 1;

            theday = thedate.getUTCDate();

        }

        else {

            theyear = thedate.getFullYear();

            themonth = thedate.getMonth() + 1;

            theday = thedate.getDate();

        }



        var monthStr = "" + themonth;

        if (monthStr < 10) {

            monthStr = "0" + monthStr;

        }

        var dayStr = "" + theday;

        if (theday < 10) {

            dayStr = "0" + dayStr;

        }



        if (theformat == "mm/dd/yy") {

            return monthStr + "/" + dayStr + "/" + theyear;

        } else

            return monthStr + "/" + dayStr;

    }



	getGroupBy(){

		return this.timegroupby.replace("proc_","");

	}

    getTimeZoneOffset() {

        if (this.useUTC) return 0;

        return -(new Date().getTimezoneOffset()) / 60

    }



    formatTitleDateString(thedate) {

        if (!(thedate instanceof Date))

            return thedate;

        if (this.titleDateFormat == "utc") {

            var isostring = thedate.toISOString();

            switch (this.getGroupBy()) {

                case "dayofyear": return isostring.substring(0, 10);

                case "dayhourofyear": return isostring;

                case "monthofyear": return isostring.substring(0, 7);

                case "yearmonthdayhour": return isostring;

                case "yearmonthday": return isostring.substring(0, 10);

                case "yearmonth": return isostring.substring(0, 7);

                default:

                    return isostring;

            }

        }

        else if (this.titleDateFormat == "unix") {

            return "" + Math.floor(thedate.getTime()/1000);

        }

        return thedate.toLocaleDateString(this.locale, this.getTitleDateOptions()).replace(/,/g, ' ');

    }





    formatDatetoLocaleDateString(thedate) {

        if (!(thedate instanceof Date)) {

            return thedate;

        }

        if (this.titleDateFormat == "utc") {

            switch (this.getGroupBy()) {

                case "dayofyear": return thedate.toISOString().substring(5, 10);

                case "dayhourofyear": return thedate.toISOString().substring(5);

                case "monthofyear": return thedate.toISOString().substring(5, 7);

                case "yearmonthdayhour": return thedate.toISOString();

                case "yearmonthday": return thedate.toISOString().substring(0, 10);

                case "yearmonth": return thedate.toISOString().substring(0, 7);

                default:

                    return thedate.toISOString();

            }

        }

        else if (this.titleDateFormat == "unix") {

            return "" + Math.floor(thedate.getTime()/1000);

        }

		return this.getFormattedLocalDate(thedate);

        //return thedate.toLocaleDateString(this.locale, this.getDateOptions());

    }





	getFormattedLocalDate(thedate){

		switch (this.getGroupBy()) {

                case "dayofyear": return this.month_strings_short[thedate.getMonth()] + "-" + thedate.getDate();

                case "dayhourofyear": 

                case "monthofyear": this.month_strings_short[thedate.getMonth()];

                case "yearmonthdayhour": return thedate.toString();

                case "yearmonthday": return this.month_strings_short[thedate.getMonth()] + " " + thedate.getDate() + " " + thedate.getFullYear();

                case "yearmonth": return this.month_strings_short[thedate.getMonth()] + " " + thedate.getFullYear();

                default:

                    return thedate.toString();

            }

	}

	

	

    getTitleDateOptions() {

        var options;

        switch (this.getGroupBy()) {

            case "dayofyear":

                options = { month: this.monthFormat, day: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            case "dayhourofyear":

                options = { month: this.monthFormat, day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            case "monthofyear":

                //options = { month: this.monthFormat, hour12: false, timeZoneName: "short" };

                options = { month: this.monthFormat, year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            case "yearmonthdayhour":

                options = { month: this.monthFormat, day: '2-digit', hour: '2-digit', minute: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            case "yearmonthday":

                options = { month: this.monthFormat, day: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            case "yearmonth":

                options = { month: this.monthFormat, year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            default:

                options = { month: this.monthFormat, day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

        }

        if (this.useUTC) {

            options.timeZone = "UTC"

        }

        return options;

    }





    getDateOptions() {

        var options;

        switch (this.getGroupBy()) {

            case "dayofyear":

                options = { month: this.monthFormat, day: '2-digit', hour12: false, timeZoneName: "short" };

                break;

            case "dayhourofyear":

                options = { month: this.monthFormat, day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false, timeZoneName: "short" };

                break;

            case "monthofyear":

                options = { month: this.monthFormat, hour12: false, timeZoneName: "short" };

                break;

            case "yearmonthdayhour":

                options = { month: this.monthFormat, day: '2-digit', hour: '2-digit', minute: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            case "yearmonthday":

                options = { month: this.monthFormat, day: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            case "yearmonth":

                options = { month: this.monthFormat, year: 'numeric', hour12: false, timeZoneName: "short" };

                break;

            default:

                options = { month: this.monthFormat, day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', year: 'numeric', hour12: false, timeZoneName: "short" };

        }

        if (this.useUTC) {

            options.timeZone = "UTC"

        }

        return options;

    }





}



















