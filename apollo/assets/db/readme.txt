This directory is intended to hold the SQLite databases to queried by the Flask server. Also contained are the following: 

1) 'sources.json': This contains a json data structure with information on databases. it consists of a list of dictionaries having the following form: 

	{	
		"id":<datasource>, 
		"ext":<database file extension>,
		"label":<human readable label>, 
		"schema":<the name of the db schema>, 
		"initial_start": "yyy-mm-dd", "initial_stop": "yyy-mm-dd"
	}

  * The datasource name (e.g., 'solar_farm') indicates the name of the database. It will be combined with the extension to form a complete file name. 
  * Thel label is just a human readable label of the database (presented to end users). 
  * The schema refers to the name of the schema file, described below. 
  * initial_start and initial_stop indicate the start and stop dates that will, by default, be chosen in the web-interface for the data source. 

2) <datasource>.json: This is the schema file for the data source. It describes the database structure(e.g., providing information on each attribute in a database table). 
   For each database, there must be a corresponding schema JSON file, but two or more databases can share the same schema file. 

The actual databases can be stored in different subdirectories. When searching for a datasource, the server will first look for a subdirectory matching the schema name. If that fails, it looks in the default db directory. And if that fails, the default data source is used. 

