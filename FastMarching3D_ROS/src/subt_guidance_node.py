#!/usr/bin/env python
import sys
import numpy as np
import guidance
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from gazebo_msgs.msg import LinkStates
from visualization_msgs.msg import Marker

class guidance_controller:
	def getPosition(self, data): # Position subscriber callback function
		# Find the index of the link_state
		if (self.link_id == -1):
			i = 0
			for name in data.name[:]:
				if name == self.name + "::" + self.name + "/base_link":
					self.link_id = i
					print("link_id = %d" % self.link_id)
				i = i+1
		# Get the link state data		
		if (self.link_id == -1):
			print('Could not find robot state information in /gazebo/link_states/')
		else:
			self.position = data.pose[self.link_id].position
			q = np.array([data.pose[self.link_id].orientation.x, \
				data.pose[self.link_id].orientation.y, data.pose[self.link_id].orientation.z, \
				data.pose[self.link_id].orientation.w])
			self.R = np.zeros((3,3))
			self.R[0,0] = q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2]
			self.R[0,1] = 2.0*(q[0]*q[1] - q[3]*q[2])
			self.R[0,2] = 2.0*(q[3]*q[1] + q[0]*q[2])

			self.R[1,0] = 2.0*(q[0]*q[1] + q[3]*q[2])
			self.R[1,1] = q[3]*q[3] - q[0]*q[0] + q[1]*q[1] - q[2]*q[2]
			self.R[1,2] = 2.0*(q[1]*q[2] - q[3]*q[0])

			self.R[2,0] = 2.0*(q[0]*q[2] - q[3]*q[1])
			self.R[2,1] = 2.0*(q[3]*q[0] + q[1]*q[2])
			self.R[2,2] = q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2]

			self.positionUpdated = 1
		return

	def getPath(self, data): # Path subscriber callback function
		newpath = np.empty((3,len(data.poses)))
		for i in range(0,len(data.poses)):
			newpath[0,i] = data.poses[i].pose.position.x
			newpath[1,i] = data.poses[i].pose.position.y
			newpath[2,i] = data.poses[i].pose.position.z
		self.path = newpath
		print('path received of size: %d' % self.path.shape[1])
		self.pathUpdated = 1
		return

	def publishLookahead(self):
		# Remove the old marker
		self.L2_marker.action = 2
		self.pub2.publish(self.L2_marker)

		# Add a new one
		self.L2_marker.action = 0
		self.L2_marker.points[0] = self.position
		self.L2_marker.points[1] = self.L2
		self.pub2.publish(self.L2_marker)
		return

	def updateCommand(self): # Updates the twist command for publishing
		# Check if the subscribers have updated the robot position and path
		if (self.path.shape[1] < 1):
			print("No guidance command, path is empty.")
			return

		# Convert the body frame x-axis of the robot to inertial frame
		heading_body = np.array([[1.0], [0.0], [0.0]]) # using commanded velocity for now (use actual later)
		heading_inertial = np.matmul(self.R, heading_body)
		velocity_inertial = self.speed*np.array([heading_inertial[0,0], heading_inertial[1,0], heading_inertial[2,0]])

		# Find the lookahead/carrot point for the guidance controller
		# Store the vehicle position for now
		p_robot = np.array([self.position.x, self.position.y, self.position.z])
		path = self.path
		start = np.array([path[0,0], path[1,0], path[2,0]])
		goal = np.array([path[0,-1], path[1,-1], path[2,-1]])
		if (self.path.shape[1] < 2):
			p_L2 = goal
			v_L2 = (goal - p_robot)/np.linalg.norm(goal - p_robot)
			print("Path is only one point long, heading to goal point.")
		else:
			p_L2, v_L2 = guidance.find_Lookahead_Discrete_3D(path, p_robot, self.speed*self.Tstar, 0, 0)

		# If p_L2 is the start of the path, check if the goal point is within an L2 radius of the vehicle, if so, go to the goal point
		if (np.linalg.norm(p_L2 - start) <= 0.05 and np.linalg.norm(p_robot - goal) <= 0.9*self.speed*self.Tstar):
			p_L2 = goal

		# Generate a lateral acceleration command from the lookahead point
		if self.controller_type == 'trajectory_shaping':
			a_cmd = guidance.trajectory_Shaping_Guidance(np.array([p_L2[0], p_L2[1]]), p_robot[0:2], \
													 np.array([velocity_inertial[0], velocity_inertial[1]]), np.array([v_L2[0], v_L2[1]]))
			chi_dot = -a_cmd/self.speed
			if (self.vehicle_type == 'air'):
				chi_dot = -chi_dot # reverse convention
		else:
			if (self.vehicle_type == 'ground'):
				a_cmd = guidance.L2_Plus_Guidance_2D(np.array([p_L2[0], p_L2[1]]), p_robot[0:2], \
													 np.array([velocity_inertial[0], velocity_inertial[1]]), self.Tstar, 0)
				chi_dot = -a_cmd/self.speed
			else:
				# a_cmd = guidance.L2_Plus_Guidance_3D(p_L2, p_robot, velocity_inertial, self.Tstar, 0)
				# Convert lateral acceleration to angular acceleration about the z axis
				# chi_dot = a_cmd[1]/self.speed
				a_cmd = guidance.L2_Plus_Guidance_2D(np.array([p_L2[0], p_L2[1]]), p_robot[0:2], \
													 np.array([velocity_inertial[0], velocity_inertial[1]]), self.Tstar, 0)
				chi_dot = -a_cmd/self.speed

		# Update class members
		self.L2.x = p_L2[0]
		self.L2.y = p_L2[1]
		self.L2.z = p_L2[2]
		L2_vec = p_L2 - p_robot
		# Change what the vehicle does depending on the path orientation relative to the robot
		dot_prod = np.dot(L2_vec[0:2], heading_inertial[0:2])/(np.linalg.norm(L2_vec[0:2])*np.linalg.norm(heading_inertial[0:2]))
		print("The heading vector in 2D is: [%0.2f, %0.2f]" % (heading_inertial[0], heading_inertial[1]))
		print("The L2 point is: [%0.2f, %0.2f]" % (p_L2[0], p_L2[1]))
		print("The robot position is : [%0.2f, %0.2f]" % (p_robot[0], p_robot[1]))
		print("The L2 vector in 2D is: [%0.2f, %0.2f]" % (L2_vec[0], L2_vec[1]))
		print("cos(eta) = %0.2f" % dot_prod)
		if (dot_prod > (.5)):
			self.command.linear.x = self.speed
			self.command.angular.z = chi_dot
		# elif (dot_prod < 0.0):
		# 	self.command.linear.x = -self.speed
		# 	self.command.angular.z = chi_dot
		else:
			self.command.linear.x = 0.0
			self.command.angular.z = chi_dot

		# Do altitude control for air vehicles
		if self.vehicle_type == 'air':
			error = L2_vec[2]
			self.command.linear.z = self.gain_z*error
		return

	def __init__(self, name='X1', vehicle_type='ground', controller_type='L2', speed=1.0):
		# Set controller specific parameters
		self.name = name; # robot name
		self.vehicle_type = vehicle_type; # vehicle type (ground vs air)
		self.controller_type = controller_type; # Type of guidance controller from guidance
		self.speed = float(speed) # m/s
		self.Tstar = 1.5 # seconds

		# Booleans for first subscription receive
		self.positionUpdated = 0
		self.pathUpdated = 0

		# Initialize ROS node and Subscribers
		node_name = self.name + '_guidance_controller'
		rospy.init_node(node_name)
		rospy.Subscriber('/gazebo/link_states', LinkStates, self.getPosition)
		self.link_id = -1
		self.path = np.empty((3,0))
		rospy.Subscriber('/' + name + '/planned_path', Path, self.getPath)

		# Initialize Publisher topics
		self.pubTopic1 = '/' + name + '/cmd_vel'
		self.pub1 = rospy.Publisher(self.pubTopic1, Twist, queue_size=10)
		self.pubTopic2 = '/' + name + '/lookahead_vec'
		self.pub2 = rospy.Publisher(self.pubTopic2, Marker, queue_size=10)

		# Initialize twist object for publishing
		self.command = Twist()
		self.command.linear.x = 0.0
		self.command.linear.y = 0.0
		self.command.linear.z = 0.0
		self.command.angular.x = 0.0
		self.command.angular.y = 0.0
		self.command.angular.z = 0.0

		# Initialize Lookahead vector for publishing
		self.L2 = Point()
		self.L2.x = 0.0
		self.L2.y = 0.0
		self.L2.z = 0.0
		self.position = Point()
		self.position.x = 0.0
		self.position.y = 0.0
		self.position.z = 0.0
		self.L2_marker = Marker()
		self.L2_marker.type = 4
		self.L2_marker.header.frame_id = "world"
		self.L2_marker.header.stamp = rospy.Time()
		self.L2_marker.id = 101;
		self.L2_marker.scale.x = 0.05
		self.L2_marker.color.b = 1.0
		self.L2_marker.color.a = 1.0
		self.L2_marker.pose.orientation.w = 1.0
		self.L2_marker.action = 0
		self.L2_marker.points.append(self.position)
		self.L2_marker.points.append(self.L2)

		# Altitude controller
		self.gain_z = 0.2
		# # Initialize velocity vector for publishing
		# self.vel_marker = Marker()
		# self.vel_marker.type = 0


	def start(self):
		rate = rospy.Rate(10.0) # 10Hz
		while not rospy.is_shutdown():
			rate.sleep()
			self.updateCommand()
			self.pub1.publish(self.command)
			self.publishLookahead()
		return


if __name__ == '__main__':
	num_args = len(sys.argv)
	# if (num_args != 4):
		# print("Node requires 4 inputs arguments")

	controller = guidance_controller(name=sys.argv[1], vehicle_type=sys.argv[2], \
									 controller_type=sys.argv[3], speed=sys.argv[4])

	try:
		controller.start()
	except rospy.ROSInterruptException:
		pass