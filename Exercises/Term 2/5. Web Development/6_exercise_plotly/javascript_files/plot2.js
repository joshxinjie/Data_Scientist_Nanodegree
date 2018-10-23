var year = [2015];
var arable_land_brazil = [33.8100342899258];
var country_name_brazil = 'Brazil';

var arable_land_germany = [47.9592041483809];
var country_name_germany = 'Germany';

var arable_land_china = [56.2229587724434];
var country_name_china = 'China';

var trace1 = {
/* TODO: Use the country name and arable land data to make a bar chart */
  x:['Brazil', 'Germany', 'China'],
  y:[arable_land_brazil[0], arable_land_germany[0], arable_land_china[0]],
  type: "bar"
};

var layout = {
 /* TODO: include a title for the chart */
  title:"Propprtion of Land <br> Used for Agriculture",
  xaxis: {
    title: "Country"
  },
  yaxis: {
    title: "Percentage of Land (%)"
  }
  
};

var data = [trace1];

Plotly.newPlot('plot2', data, layout);