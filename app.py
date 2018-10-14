from flask import Flask
from flask import request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import copy
import json as json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello():


	class Vector2(object):
	  x = None
	  y = None
	  
	  def __init__(self, x, y):
	    self.x = x
	    self.y = y 
	  def getJson(self):
	    data = {}
	    data['x'] = self.x
	    data['y'] = self.y
	    json_data = json.dumps(data)
	    return json_data
	  
	class LayoutObject(object):
	  empty = 0
	  
	  def getIntersection(self, otherLayoutObject):
	    return None #not implemented
	 
	  def __copy__(self):
	    print("ERROR NOT IMPLEMENTED")
	  
	class Box(LayoutObject):
	  center = None
	  size = None
	  uL = None
	  uR = None
	  bL = None
	  bR = None

	  #Center is a Vector2
	  def __init__(self, center, size):
	    self.center = center
	    self.size = size
	    self.uL = Vector2(center.x - size/2., center.y + size/2)
	    self.uR = Vector2(center.x + size/2., center.y + size/2) 
	    self.bL = Vector2(center.x - size/2., center.y - size/2)    
	    self.bR = Vector2(center.x + size/2., center.y - size/2) 
	  
	  def __copy__(self):
	    return Box(self.center, self.size)
	  
	  #Only supports intersections with other boxes
	  def getIntersection(self, otherLayoutObject):
	    # determine the (x, y)-coordinates of the intersection rectangle
	    xA = max(self.uL.x, otherLayoutObject.uL.x)
	    yA = min(self.uL.y, otherLayoutObject.uL.y) #A : upper left
	    xB = min(self.bR.x, otherLayoutObject.bR.x)
	    yB = max(self.bR.y, otherLayoutObject.bR.y) #B: bottom right
	    return max(0, xB - xA) * max(0, yA - yB)
	  
	  def getJson(self):
	    data = {}
	    data['center'] = self.center.getJson()
	    data['size'] = self.size
	    json_data = json.dumps(data)
	    return json_data
	    
	class Layout(object):
	  
	  def __init__(self, layoutObjects, planeBox):
	    self._myLayoutObjects = layoutObjects
	    self.planeBox = planeBox
	  
	  def getLayoutObjects(self):
	    return self._myLayoutObjects
	  
	  def getPlaneBox(self):
	    return self.planeBox
	  
	  def drawLayout(self):
	    plt.axes()
	    x = np.arange(len(self._myLayoutObjects))
	    ys = [i+x+(i*x)**2 for i in range(len(self._myLayoutObjects))]
	    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
	    #For now all layout objects are Boxes
	    i = 0
	    for box in self._myLayoutObjects:
	      shape = plt.Rectangle((box.bL.x, box.bL.y), box.size, box.size, fc=colors[i])
	      plt.gca().add_patch(shape)
	      #label the box
	      plt.text(box.center.x, box.center.y, str(i))
	      i += 1
	    #bounding box
	    shape = plt.Rectangle((self.planeBox.bL.x, self.planeBox.bL.y), self.planeBox.size, self.planeBox.size, fc="c", alpha=0.1)
	    plt.gca().add_patch(shape)
	    plt.axis('scaled')
	    plt.show()
	   
	  def __copy__(self):
	    newLayoutObjects = []
	    for layoutObject in self.getLayoutObjects():
	      newLayoutObject = copy(layoutObject)
	      newLayoutObjects.append(newLayoutObject)
	    return Layout(*newLayoutObjects) 
	  def getJson(self):
	    ##returns a json represetnation of plane box and layout objects
	    data = {}
	    layoutObjsJson = []
	    layoutObjects = self.getLayoutObjects()
	    for obj in layoutObjects:
	      layoutObjsJson.append(obj.getJson())
	      data['layoutObjects'] = layoutObjsJson
	    data['planeBox'] = self.planeBox.getJson()
	    return data


	class Constraint(object):
	  
	  def evaluate_cost(self):
	    return 0

	class AreaConstraint(Constraint):
	  
	  def __init__(self, object_indices):
	    self._object_indices = object_indices
	  
	  def evaluate_cost(self, layoutObjects):
	    sumOfAreas = 0
	    for i in range(len(self._object_indices)):
	      for j in range(i + 1, len(self._object_indices)):
	        sumOfAreas += layoutObjects[self._object_indices[i]].getIntersection(layoutObjects[self._object_indices[j]])
	    return sumOfAreas

	class BoundaryConstraint(Constraint):
	  
	  def __init__(self, bbox):
	    self.bbox = bbox
	    
	  def evaluate_cost(self, layoutObjects):
	    for layoutObject in layoutObjects:
	      if not(layoutObject.bR.x < self.bbox.bR.x and layoutObject.bL.x > self.bbox.bL.x and layoutObject.uR.y < self.bbox.uR.y and layoutObject.bR.y > self.bbox.bR.y):
	        return float("inf")
	    return 0

	  
	class DistanceConstraint(Constraint):
	  """
	  Constraint for distance between all objects in layoutObjects within 
	  some threshold. 
	  """
	  
	  def __init__(self, object_indices, threshold):
	    self._object_indices = object_indices
	    self._threshold = threshold
	  
	  def evaluate_cost(self, layoutObjects):
	    cost = 0
	    
	    for i in range(len(self._object_indices)):
	      for j in range(i + 1, len(self._object_indices)):
	        distance = np.sqrt((layoutObjects[self._object_indices[i]].center.x - layoutObjects[self._object_indices[j]].center.x)**2 + 
	                          (layoutObjects[self._object_indices[i]].center.y - layoutObjects[self._object_indices[j]].center.y)**2)
	        if distance >= self._threshold:
	          cost += distance **2
	    
	    return cost

	def boltzmann(cost, beta=1.0):
	  return np.exp(-beta*cost)

	def acceptanceProbability(current_cost, proposed_cost):
	  return min(1, boltzmann(proposed_cost)/boltzmann(current_cost))


	def get_cost(layout, constraints):
	  layoutObjects = layout.getLayoutObjects()
	  cost = 0

	  for constraint in constraints:
	    cost += constraint.evaluate_cost(layoutObjects)

	  return cost

	  # areaConstraintObj = AreaConstraint(layout.getLayoutObjects())
	  # boundaryConstraintObj = BoundaryConstraint(layout.getLayoutObjects(), layout.getPlaneBox())
	  # distanceConstraintObj = DistanceConstraint(layout.getLayoutObjects()[:2], 0.6)
	  # return areaConstraintObj.evaluate_cost() + boundaryConstraintObj.evaluate_cost() #+ distanceConstraintObj.evaluate_cost()


	def propose_new_layout(current_layout):
	  new_layout_objects = []
	  for rect in current_layout.getLayoutObjects():
	    rect_center = rect.center 
	    new_center = Vector2(rect_center.x + np.random.normal(scale=0.01), rect_center.y + np.random.normal(scale=0.01))
	    new_rect = Box(new_center, rect.size)
	    new_layout_objects.append(new_rect)
	  
	  return Layout(new_layout_objects, current_layout.getPlaneBox())


	def metropolis_hastings(initial_layout, constraints, num_iters=10000):
	  cur_cost = get_cost(initial_layout, constraints)
	  cur_layout = initial_layout
	  best_layout = cur_layout
	  best_cost = float("inf")
	  
	  for i in range(num_iters):
	    new_layout = propose_new_layout(cur_layout)
	    new_cost = get_cost(new_layout, constraints)
	    
	    if np.random.random() < acceptanceProbability(cur_cost, new_cost):
	      cur_layout = new_layout
	      cur_cost = new_cost
	      if cur_cost < best_cost:
	        best_layout = cur_layout
	        best_cost = cur_cost
	      
	  return best_layout

	def deserialize(data):
	  layout_objects = []
	  constraints = []

	  for obj_data in data['layout']['layoutObjects']: 
	    center = Vector2(obj_data['center']['x'], obj_data['center']['y'])
	    size = obj_data['size']
	    layout_objects.append(Box(center, size))
	  
	  # Bounding Box
	  bbox_data = data['layout']['planeBox']
	  bbox_center = Vector2(bbox_data['center']['x'], bbox_data['center']['y'])
	  bbox_size = bbox_data['size']
	  bbox = Box(bbox_center, bbox_size)

	  # Constraints
	  for name in data['constraints']:

	    if name == "areaConstraints":
	      for param_info in data['constraints'][name]:
	        constraints.append(AreaConstraint(param_info['_object_indices']))

	    elif name == "boundaryConstraints":
	      # import pdb; pdb.set_trace()

	      for param_info in data['constraints'][name]:
	        constraints.append(BoundaryConstraint(bbox))

	    elif name == "distanceConstraints":
	      for param_info in data['constraints'][name]:
	        constraints.append(DistanceConstraint(param_info['_object_indices'], param_info['threshold']))

	  
	  layout = Layout(layout_objects, bbox)
	  return layout, constraints




	# data = {"layout":{"layoutObjects":[{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0},{"center":{"x":0.0,"y":0.0},"size":1.0}],"planeBox":{"center":{"x":0.0,"y":0.0},"size":10.0}},"constraints":{"areaConstraints":[{"_object_indices":[0,1,2,3,4,5,6,7,8,9]}],"boundaryConstraints":[]}}
	incoming_data = request.get_json()
	# json_object = json.dumps(incoming_data)
	data = incoming_data
	print(data)

	# print(json_object)
	layout, constraints = deserialize(data)
	# layout.drawLayout()

	final_layout = metropolis_hastings(layout, constraints, num_iters=100000)

	# final_layout.drawLayout()
	final_layout.getJson()

	return json.dumps(final_layout.getJson())


if __name__ == '__main__':
    app.run(debug=True)