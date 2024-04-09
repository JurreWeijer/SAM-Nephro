from shapely import Polygon, MultiPolygon
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def geojson_to_polygon(
    data: dict, 
    classes: list = None, 
    without_holes: bool = False, 
    without_multi: bool = False):
    """
    Converts a GeoJSON file to a list of dictionaries containing Shapely polygons and the corresponding label.

    Args:
        data (dict): Information loaded from a GeoJSON file.
        classes (list, optional): The classes that have to be extracted from the JSON data. If set to None, all polygons are extracted.
        without_holes (bool, optional): Determines if the holes within the polygons are kept or removed. Set to True if the holes have to be removed from polygons that have them.
        without_multi (bool, optional): Determines if the multipolygons are added to the returned list or if they are skipped. Set to True if the multipolygons have to be skipped.

    Returns:
        list: A list of dictionaries, each containing the polygons under the key 'polygon' and the label under the key 'label'.
    """

    polygons = []

    # Loop over all the polygons
    for feature in data["features"]:
        new_polygon = {}

        # Extract label from properties
        if "classification" in feature["properties"]:
            new_polygon["label"] = feature["properties"]["classification"]["name"]
        else:
            new_polygon["label"] = "no label"

        # Skip if classes are specified and the label is not in the list
        if classes is not None and new_polygon["label"] not in classes:
            continue

        # Extract coordinates and type from geometry
        polygon = feature['geometry']['coordinates']
        type = feature['geometry']['type']

        # Interior rings are the holes and as a standard set to None
        interior_rings = None

        # Handle MultiPolygons
        if type == "MultiPolygon":
            if without_multi:
                continue

            multipolygon = []
            # Handle the multiple polygons in the MultiPolygon
            for poly in polygon:
                
                # Handle MultiPolygons with holes
                if len(poly) > 1:
                    exterior_ring = poly[0]
                    exterior_ring.append(exterior_ring[0])

                    # If without holes the polygons for the holes are skipped
                    if not without_holes:
                        interior_rings = []
                        for interior_ring in poly[1:]:
                            interior_ring.append(interior_ring[0])
                            interior_rings.append(interior_ring)

                    multipolygon.append(Polygon(exterior_ring, interior_rings))

                # Handle MultiPolygons without holes
                elif len(poly) == 1:
                    exterior_ring = poly[0]
                    exterior_ring.append(exterior_ring[0])

                    multipolygon.append(Polygon(exterior_ring))

            new_polygon["polygon"] = MultiPolygon(multipolygon)

        # Handle Polygons
        elif type == "Polygon":
            # Handle Polygons with holes
            if len(polygon) > 1:
                exterior_ring = polygon[0]
                exterior_ring.append(exterior_ring[0])

                if not without_holes:
                    interior_rings = []
                    for interior_ring in polygon[1:]:
                        interior_ring.append(interior_ring[0])
                        interior_rings.append(interior_ring)

            # Handle Polygons without holes
            elif len(polygon) == 1:
                exterior_ring = polygon[0]
                exterior_ring.append(exterior_ring[0])

            new_polygon["polygon"] = Polygon(exterior_ring, interior_rings)
                
        # Check for duplicate polygons
        duplicate = False
        for existing_polygon in polygons:
            if existing_polygon["polygon"].equals(new_polygon["polygon"]) and existing_polygon["label"] == new_polygon["label"]:
                duplicate = True
        
        if without_multi: 
            if not isinstance(new_polygon["polygon"], Polygon): 
                duplicate = True

        # Append to the list if not a duplicate
        if not duplicate: #and isinstance(new_polygon["polygon"], Polygon):
            polygons.append(new_polygon)

    return polygons
    
def labeled_polygons_to_geojson(polygons, level, with_holes=True):
    """converts a a list of polygons to a geojson file format

    Args:
        polygons: a list of polygons for a certain slide
        mask_level: the level at which the mask was created

    Returns:
        feature_collection: a collection of all the annotations in the WSI
    """
    
    features = []

    for ids, polygon_dict in enumerate(polygons):
        
        if isinstance(polygon_dict["polygon"], Polygon):
            polygon = list(polygon_dict["polygon"].exterior.coords)
            polygon = [[x * (2 ** level), y * (2 ** level)] for x, y in polygon] 
            polygon.append(polygon[0])  # Close the polygon by repeating the first point
            current_type = "Polygon"
            
            if with_holes:
                interior_coords = []
                for interior_ring in polygon_dict["polygon"].interiors:
                    coords = interior_ring.coords[:]
                    coords = [[x * (2 ** level), y * (2 ** level)] for x, y in coords]
                    coords.append(coords[0])  # Close the interior ring
                    interior_coords.append(coords)

                polygon = [polygon] + interior_coords 

        elif isinstance(polygon_dict["polygon"], MultiPolygon):
            multipolygon = polygon_dict["polygon"]
            polygons_list = []
            
            for polygon in multipolygon.geoms:
                polygon_coords = list(polygon.exterior.coords)
                polygon_coords = [[x * (2 ** level), y * (2 ** level)] for x, y in polygon_coords]
                polygon_coords.append(polygon_coords[0])
                
                if with_holes:
                    interior_coords = []
                    for interior_ring in polygon.interiors:
                        coords = interior_ring.coords[:]
                        coords = [[x * (2 ** level), y * (2 ** level)] for x, y in coords]
                        coords.append(coords[0])  # Close the interior ring
                        interior_coords.append(coords)

                    polygon_coords = [polygon_coords] + interior_coords
                
                polygons_list.append(polygon_coords) 
                
            polygon = polygons_list
            current_type = "MultiPolygon"
            
        label = polygon_dict["label"]

        feature = {
            "type": "Feature",
            "id": ids,
            "geometry": {
                "type": current_type,
                "coordinates": polygon
            },
            "properties": {
                "objectType": "annotation",
                "classification" : {
                    "name": str(label),
                    "color": [0,0,0]
                }
            }
        }

        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    return feature_collection

def polygons_to_geojson(polygons, level):
    """converts a a list of polygons to a geojson file format

    Args:
        polygons: a list of polygons for a certain slide
        mask_level: the level at which the mask was created

    Returns:
        feature_collection: a collection of all the annotations in the WSI
    """
    
    features = []

    for polygon_dict in polygons:
        polygon = polygon_dict["polygon"].exterior.coords[:]
        polygon = [[x * (2 ** level), y * (2 ** level)] for x, y in polygon]
        polygon.append(polygon[0])  # Close the polygon by repeating the first point

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                "objectType": "annotation",
            }
        }

        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    return feature_collection

def geojson_to_mask(data, slide, mask_level, set_labels = None):
    """converts a geojson file to a binary mask for a WSI

    Args:
        data: information loaded from a geojson file 
        slide: the corresponding slide for the geojson file 
        mask_level: the level at which you want the mask to be
        set_labels: optional set of labels which you want to retrieve from the file, if none all labels will be retreived
    
    Returns:
        mask: a binary mask of the WSI
        mask_info: a dictionary providing information of what channel contains what label
    """
    
    if set_labels is None:
        labels = {}
    else:
        labels = set_labels

    for feature in data["features"]:
        if "classification" in feature["properties"]:
            label = feature["properties"]["classification"]["name"]
        else:
            label = "no-label"
        
        polygon = feature['geometry']['coordinates']
        type= feature['geometry']['type']
        
        #find the multipolygons 
        if type == "MultiPolygon":
            #loop over the multiple polygons in the multipolygon
            for poly in polygon:
                scaled_polygon = []
                for coordinate_set in poly:
                    coordinate_set.append(coordinate_set[0])
                    scaled_set = (np.array(coordinate_set) / (2 ** mask_level)).astype(np.int32)
                    scaled_polygon.append(scaled_set)   
                    
                    if label in labels: 
                        labels[label].append(scaled_polygon)
                    elif label not in labels and set_labels is None: 
                        labels[label] = []
                        labels[label].append(scaled_polygon) 
                    
        #find the polygons                      
        elif type == "Polygon": 
            scaled_polygon = []
            for coordinate_set in polygon:
                coordinate_set.append(coordinate_set[0])
                scaled_set = (np.array(coordinate_set) / (2 ** mask_level)).astype(np.int32)
                scaled_polygon.append(scaled_set)
                
                
            if label in labels: 
                labels[label].append(scaled_polygon)
            elif label not in labels and set_labels is None: 
                labels[label] = []
                labels[label].append(scaled_polygon)
            
        
    channels = []
    mask_info = {}
    background = np.zeros((slide.level_dimensions[mask_level][1],slide.level_dimensions[mask_level][0]), dtype=np.uint8)
    for i, label in enumerate(labels, start=1):
        blank = np.zeros((slide.level_dimensions[mask_level][1],slide.level_dimensions[mask_level][0]), dtype=np.uint8)
        
        #cv2.fillPoly
        for polygon in labels[label]:
            cv2.drawContours(blank, polygon, -1, i, thickness=cv2.FILLED)
        
        #_ , binary_blank = cv2.threshold(blank, 1, i, cv2.THRESH_BINARY)
        channels.append(blank)
                
        for polygon in labels[label]:
            cv2.drawContours(background, polygon, -1, 255, thickness=cv2.FILLED)
        mask_info[label] = i 
    
    _, binary_background = cv2.threshold(background, 1, 255, cv2.THRESH_BINARY)
    _, inv_background = cv2.threshold(binary_background, 127, 255, cv2.THRESH_BINARY_INV)
    channels.insert(0, inv_background)
                
    mask = np.stack(channels, axis = 2) 
    mask = mask.transpose(1,0,2)

    return mask, mask_info

def geojson_to_mask_test(data, slide, mask_level, with_border, set_labels = None):
    """converts a geojson file to a binary mask for a WSI

    Args:
        data: information loaded from a geojson file 
        slide: the corresponding slide for the geojson file 
        mask_level: the level at which you want the mask to be
        set_labels: optional set of labels which you want to retrieve from the file, if none all labels will be retreived
    
    Returns:
        mask: a binary mask of the WSI
        mask_info: a dictionary providing information of what channel contains what label
    """
    
    if set_labels is None:
        labels = {}
    else:
        labels = set_labels

    for feature in data["features"]:
        if "classification" in feature["properties"]:
            label = feature["properties"]["classification"]["name"]
        else:
            label = "no-label"
        
        polygon = feature['geometry']['coordinates']
        type= feature['geometry']['type']
        
        #find the multipolygons 
        if type == "MultiPolygon":
            #loop over the multiple polygons in the multipolygon
            for poly in polygon:
                scaled_polygon = []
                for coordinate_set in poly:
                    coordinate_set.append(coordinate_set[0])
                    scaled_set = (np.array(coordinate_set) / (2 ** mask_level)).astype(np.int32)
                    scaled_polygon.append(scaled_set)   
                    
                    if label in labels: 
                        labels[label].append(scaled_polygon)
                    elif label not in labels and set_labels is None: 
                        labels[label] = []
                        labels[label].append(scaled_polygon) 
                    
        #find the polygons                      
        elif type == "Polygon": 
            scaled_polygon = []
            for coordinate_set in polygon:
                coordinate_set.append(coordinate_set[0])
                scaled_set = (np.array(coordinate_set) / (2 ** mask_level)).astype(np.int32)
                scaled_polygon.append(scaled_set)
                
                
            if label in labels: 
                labels[label].append(scaled_polygon)
            elif label not in labels and set_labels is None: 
                labels[label] = []
                labels[label].append(scaled_polygon)
            
        
    channels = []
    background = np.zeros((slide.level_dimensions[mask_level][1],slide.level_dimensions[mask_level][0]), dtype=np.uint8)
    if with_border:
        border_channel = np.zeros((slide.level_dimensions[mask_level][1],slide.level_dimensions[mask_level][0]), dtype=np.uint8)
    for i, label in enumerate(labels, start=1):
        blank = np.zeros((slide.level_dimensions[mask_level][1],slide.level_dimensions[mask_level][0]), dtype=np.uint8)
        
        #cv2.fillPoly
        for polygon in labels[label]:
            cv2.drawContours(blank, polygon, -1, i, thickness=cv2.FILLED)
        
        #_ , binary_blank = cv2.threshold(blank, 1, i, cv2.THRESH_BINARY)
        channels.append(blank)
                
        for polygon in labels[label]:
            cv2.drawContours(background, polygon, -1, 255, thickness=cv2.FILLED)
        
        if with_border:
            border_value = len(labels) + 1
            for polygon in labels[label]:
                cv2.drawContours(border_channel, polygon, -1, border_value, thickness=4)
    
    _, binary_background = cv2.threshold(background, 1, 255, cv2.THRESH_BINARY)
    _, inv_background = cv2.threshold(binary_background, 127, 255, cv2.THRESH_BINARY_INV)
    channels.insert(0, inv_background)
    if with_border:
        channels.insert(1, border_channel)
                
    mask = np.stack(channels, axis = 2) 
    mask = mask.transpose(1,0,2)
    mask = np.argmax(mask, axis=2)

    return mask

