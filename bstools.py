import numpy as np

def get_lat_lon_from_osm_route(G, route):
    """
    Get lat and lon from an osmnx route (list of nodes) and nx.MultGraph.
    The function returns two numpy arrays with the lat and lon of route.
    Also return a GeoDataFrame with the lat and lon of the route as a
    linestring.
    Parameters
    ----------
    G : nx.MultiGraph
        Graph to get lat and lon from. Graph should be built
        with osmnx.get_undirected.
    route : list
        List of nodes to build edge and to get lat lon from
    Returns
    -------
    lat : numpy.ndarray
        Array with latitudes of route
    lon : numpy.ndarray
        Array with longitudes of route
    """
    # add first node to route
    lons = np.array(G.nodes[route[0]]["x"])
    lats = np.array(G.nodes[route[0]]["y"])

    # loop through the rest for loop only adds from second point of edge
    for u, v in zip(route[:-1], route[1:]):
        # if there are parallel edges, select the shortest in length
        data = list(G.get_edge_data(u, v).values())[0]

        # extract coords from linestring
        xs, ys = data["geometry"].xy

        # Check if geometry of edge is in correct order
        if G.nodes[u]["x"] != data["geometry"].coords[0][0]:

            # flip if in wrong order
            xs = xs[::-1]
            ys = ys[::-1]

        # only add from the second point of linestring
        lons = np.append(lons, xs[1:])
        lats = np.append(lats, ys[1:])

    return lats, lons

def get_turn_arrays(lats, lons, cutoff_angle=25):
    """
    Get turn arrays from latitude and longitude arrays.
    The function returns the turn boolean for cases 
    when angle is larger than the cutoff.
    Parameters
    ----------
    lat : numpy.ndarray
        Array with latitudes of route
    lon : numpy.ndarray
        Array with longitudes of route
    cutoff_angle : int
        Cutoff angle for turning. Default is 25.
    Returns
    -------
    turn_bool : numpy.ndarray
        Array with boolean values for turns.
    """

    # Define empty arrays that are same size as lat and lon
    turn_bool = np.array([False] * len(lats), dtype=np.bool8)

    # Initialize variables for the loop
    lat_prev = lats[0]
    lon_prev = lons[0]

    # loop thru the points to calculate turn angles
    for i in range(1, len(lats) - 1):
        # reset some values for the loop
        lat_cur = lats[i]
        lon_cur = lons[i]
        lat_next = lats[i + 1]
        lon_next = lons[i + 1]

        # calculate angle between points
        d1 = qdrdist(lat_prev, lon_prev, lat_cur, lon_cur)
        d2 = qdrdist(lat_cur, lon_cur, lat_next, lon_next)

        # fix angles that are larger than 180 degrees
        angle = abs(d2 - d1)
        angle = 360 - angle if angle > 180 else angle

        # give the turn speeds based on the angle
        if angle > cutoff_angle and i != 0:

            # set turn bool to True
            turn_bool[i] = True

        # update the previous values at the end of the loop
        lat_prev = lat_cur
        lon_prev = lon_cur

    # make last entry to turn bool true (to slow down in time)
    turn_bool[-1] = True

    return turn_bool

def qdrdist(latd1, lond1, latd2, lond2):
    """Calculate bearing, using WGS'84
    In:
        latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
    Out:
        qdr [deg] = heading from 1 to 2
    Function is from bluesky (geo.py)
    """

    # Convert to radians
    lat1 = np.radians(latd1)
    lon1 = np.radians(lond1)
    lat2 = np.radians(latd2)
    lon2 = np.radians(lond2)

    # Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html
    coslat1 = np.cos(lat1)
    coslat2 = np.cos(lat2)

    qdr = np.degrees(
        np.arctan2(
            np.sin(lon2 - lon1) * coslat2,
            coslat1 * np.sin(lat2) - np.sin(lat1) * coslat2 * np.cos(lon2 - lon1),
        )
    )

    return qdr


def rwgs84(latd):
    """Calculate the earths radius with WGS'84 geoid definition
    In:  lat [deg] (latitude)
    Out: R   [m]   (earth radius)
    Function is from bluesky (geo.py)
    """
    lat = np.radians(latd)
    a = 6378137.0  # [m] Major semi-axis WGS-84
    b = 6356752.314245  # [m] Minor semi-axis WGS-84
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    an = a * a * coslat
    bn = b * b * sinlat
    ad = a * coslat
    bd = b * sinlat

    # Calculate radius in meters
    r = np.sqrt((an * an + bn * bn) / (ad * ad + bd * bd))

    return r


def create_scenario_text(lats, lons, turn_bool):
    """
    Creates the scenario text for a given aircraft.
    Parameters
    ----------
    acidx : int
        The aircraft index.
    lats : numpy.ndarray
        The latitudes of the path.
    lons : numpy.ndarray
        The longitudes of the path.
    turn_bool : numpy.ndarray
        A boolean array indicating whether the
        aircraft turns at a given waypoint.
    Returns
    -------
    scenario_lines : str
        The scenario text.
    """
    # bearing of the first segment
    achdg = qdrdist(lats[0], lons[0], lats[1], lons[1])

    # start a list of strings
    scenario_lines = []

    # Start writing lines
    scenario_lines.append(
      "# Allow BlueSky to understand 0 speed commands\n"
      "00:00:00>CASMACHTHR 0.0\n\n"
      "# Zoom and turn on tiledmap\n"
      "00:00:00>ZOOM 350\n"
      "00:00:00>VIS MAP TILEDMAP\n"
    )

    # Create the aircraft
    scenario_lines.append(
        "# Create drone at height 0 speed 0 and pan to it\n"
        f"00:00:00>CRE D1 M600 {lats[0]} {lons[0]} {achdg} 0 0\n"
        "00:00:00>PAN D1\n"
    )

    # Give an altitude command to start take-off
    scenario_lines.append(
        f"# Tell drone to climb to 60 feet\n"
         "00:00:00>ALT D1 60\n"
    )

    # after climbing to 60 feet give it a speed of 30 knots
    scenario_lines.append(
        f"# At 60 feet drone should increase speed to 30 knots\n"
         "00:00:00>D1 ATALT 60 SPD D1 30\n"
    )

    # Create the add waypoints command
    addwypoint_lines = ["# Add waypoints\n"
                        "00:00:00>ADDWAYPOINTS D1"
                        ]
    # add the rest of the lines as waypoints
    for i in range(1, len(lats)):
        if turn_bool[i]:
            addwypoint_lines.append(f"{lats[i]} {lons[i]},,30,TURNSPD,10")
        else:
            addwypoint_lines.append(f"{lats[i]} {lons[i]},,30,FLYBY, 0")

    # add the last waypoint twice to mark it as flyby
    addwypoint_lines.append(f"{lats[-1]} {lons[-1]},,30,FLYBY, 0\n")

    # join into one string
    scenario_lines.append(",".join(addwypoint_lines))

    # turn vnav and lnav on
    scenario_lines.append(f"# Tell drone to start following route when it is at 60 ft\n"
                           "00:00:00>D1 ATALT 60 LNAV D1 ON\n"
                           "00:00:00>D1 ATALT 60 VNAV D1 ON\n")


    # Tell aircraft to descend in last waypoint
    scenario_lines.append(
        f"# Tell drone to reach speed 0 at last waypoint and begin descending\n"
        f"00:00:00>D1 ATDIST {lats[-1]} {lons[-1]} 0.01 SPD D1 0\n"
        f"00:00:00>D1 ATDIST {lats[-1]} {lons[-1]} 0.01 D1 ATSPD 0 ALT D1 0\n"
        f"00:00:00>D1 ATDIST {lats[-1]} {lons[-1]} 0.01 D1 ATALT 0 DEL D1"
    )
 
    scenario = "\n".join(scenario_lines)

    return scenario

