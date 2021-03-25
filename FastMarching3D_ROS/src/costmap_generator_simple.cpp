// This node fuses occupancy grid Octomaps, ESDF/TSDF PointCloud2's, and traversability Octomaps into one map for robust planning and control.
// C++ Standard Libraries
#include <math.h>
// ROS libraries
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
// pcl
#include <pcl/search/kdtree.h>
#include <pcl/point_types.h>
// Octomap libaries
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
// pcl ROS
#include <pcl_conversions/pcl_conversions.h>

#define PI 3.14159265

octomap::OcTree* occupancy_tree; // OcTree object for holding occupancy Octomap
bool updated_occupancy = false;
bool updated_occupancy_first = false;
pcl::PointCloud<pcl::PointXYZI>::Ptr esdf_cloud (new pcl::PointCloud<pcl::PointXYZI>);
std_msgs::Header cost_map_header;
bool updated_ESDF = false;
bool updated_ESDF_first = false;
float octomap_free_cost = 0.0;
float unseen_cost = -1.0;

void index3_xyz(const int index, float point[3], float min[3], int size[3], float voxelSize)
{
  // x+y*sizx+z*sizx*sizy
  point[2] = min[2] + (index/(size[1]*size[0]))*voxelSize;
  point[1] = min[1] + ((index % (size[1]*size[0]))/size[0])*voxelSize;
  point[0] = min[0] + ((index % (size[1]*size[0])) % size[0])*voxelSize;
}

int xyz_index3(const float point[3], float min[3], int size[3], float voxelSize)
{
  int ind[3];
  for (int i=0; i<3; i++) ind[i] = round((point[i]-min[i])/voxelSize);
  return (ind[0] + ind[1]*size[0] + ind[2]*size[0]*size[1]);
}

void callbackOctomapBinary(const octomap_msgs::Octomap::ConstPtr msg)
{
  ROS_INFO("Octomap binary callback called");
  if (msg->data.size() == 0) return;
  delete occupancy_tree;
  occupancy_tree = new octomap::OcTree(msg->resolution);
  occupancy_tree = (octomap::OcTree*)octomap_msgs::binaryMsgToMap(*msg);
  updated_occupancy = true;
  updated_occupancy_first = true;
  ROS_INFO("Callback success!");
  return;
}

void callbackOctomapFull(const octomap_msgs::Octomap::ConstPtr msg)
{
  ROS_INFO("Octomap full callback called");
  if (msg->data.size() == 0) return;
  delete occupancy_tree;
  occupancy_tree = new octomap::OcTree(msg->resolution);
  occupancy_tree = (octomap::OcTree*)octomap_msgs::fullMsgToMap(*msg);
  updated_occupancy = true;
  updated_occupancy_first = true;
  ROS_INFO("Callback success!");
  return;
}

void CallbackESDF(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  ROS_INFO("ESDF callback called");
  if (msg->data.size() == 0) return;
  // Convert from ROS PC2 msg to PCL object
  pcl::fromROSMsg(*msg, *esdf_cloud);
  updated_ESDF = true;
  updated_ESDF_first = true;
  cost_map_header = msg->header;
  ROS_INFO("ESDF callback success!");
  return;
}

void GetPointCloudBounds(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float cloud_min[3], float cloud_max[3])
{
  // Get the xyz extents of the PCL by running one loop through the data
  if (cloud->points.size() == 0) return;
  pcl::PointXYZI p = cloud->points[0];
  cloud_min[0] = p.x; cloud_min[1] = p.y; cloud_min[2] = p.z;
  cloud_max[0] = p.x; cloud_max[1] = p.y; cloud_max[2] = p.z;
  for (int i=0; i<cloud->points.size(); i++) {
    p = cloud->points[i];
    cloud_min[0] = std::min(cloud_min[0], p.x);
    cloud_min[1] = std::min(cloud_min[1], p.y);
    cloud_min[2] = std::min(cloud_min[2], p.z);
    cloud_max[0] = std::max(cloud_max[0], p.x);
    cloud_max[1] = std::max(cloud_max[1], p.y);
    cloud_max[2] = std::max(cloud_max[2], p.z);
  }
  return;
}

void GetOcTreeBounds(octomap::OcTree* map, float tree_min[3], float tree_max[3])
{
  double x, y, z;
  float voxel_size = (float)map->getResolution();
  map->getMetricMin(x, y, z);
  tree_min[0] = (float)x - 1.5*voxel_size;
  tree_min[1] = (float)y - 1.5*voxel_size;
  tree_min[2] = (float)z - 1.5*voxel_size;
  map->getMetricMax(x, y, z);
  tree_max[0] = (float)x + 1.5*voxel_size;
  tree_max[1] = (float)y + 1.5*voxel_size;
  tree_max[2] = (float)z + 1.5*voxel_size;
  return;
}

void ConvertPointCloudToMatrix(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::vector<float> &M, float mat_min[3], float mat_max[3], float voxel_size, float default_value = -1.0)
{
  M.clear();
  int size[3];
  for (int i=0; i<3; i++) size[i] = round((mat_max[i] - mat_min[i])/voxel_size) + 1;
  M.resize(size[0]*size[1]*size[2]);
  std::fill(M.begin(), M.end(), default_value);
  for (int i=0; i<cloud->points.size(); i++) {
    pcl::PointXYZI p = cloud->points[i];
    float query[3] = {p.x, p.y, p.z};
    M[xyz_index3(query, mat_min, size, voxel_size)] = p.intensity;
  }
  return;
}

void ConvertOcTreeToMatrix(octomap::OcTree* map, std::vector<float> &M, float mat_min[3], float mat_max[3], float default_value=0.5)
{
  M.clear();
  int size[3];
  float voxel_size = (float)map->getResolution();
  for (int i=0; i<3; i++) size[i] = round((mat_max[i] - mat_min[i])/voxel_size) + 1;
  M.resize(size[0]*size[1]*size[2]);
  std::fill(M.begin(), M.end(), default_value);
  map->expand();
  for(octomap::OcTree::leaf_iterator it = map->begin_leafs(), end=map->end_leafs(); it!=end; ++it) {
    float query[3] = {(float)it.getX(), (float)it.getY(), (float)it.getZ()};
    M[xyz_index3(query, mat_min, size, voxel_size)] = it->getOccupancy();
  }
  return;
}

sensor_msgs::PointCloud2 FuseMaps()
{
  updated_ESDF = false;
  updated_occupancy = false;
  float voxel_size = (float)occupancy_tree->getResolution();

  // Get extents of both sets of map data
  float esdf_min[3], esdf_max[3], octomap_min[3], octomap_max[3], cost_map_min[3], cost_map_max[3];
  GetPointCloudBounds(esdf_cloud, esdf_min, esdf_max);
  GetOcTreeBounds(occupancy_tree, octomap_min, octomap_max);
  
  for (int i=0; i<3; i++) {
    cost_map_min[i] = std::min(esdf_min[i], octomap_min[i]);
    cost_map_max[i] = std::max(esdf_max[i], octomap_max[i]);
  }

  // Make two flat matrixes big enough for both sets of data
  std::vector<float> esdf;
  ConvertPointCloudToMatrix(esdf_cloud, esdf, cost_map_min, cost_map_max, voxel_size);
  std::vector<float> occupancy_grid;
  ConvertOcTreeToMatrix(occupancy_tree, occupancy_grid, cost_map_min, cost_map_max);

  // Combine them into the cost map
  std::vector<float> cost_map;
  cost_map.resize(esdf.size());

  for (int i=0; i<cost_map.size(); i++) {
    // 3 cases
    cost_map[i] = (esdf[i] > voxel_size)*esdf[i] // Case 1: Traversable and positive distance from obstacle
                  + ((esdf[i] == -1.0)*(occupancy_grid[i] == 0.5))*unseen_cost // Case 2: Unseen by both maps
                  + ((esdf[i] == -1.0)*(occupancy_grid[i] < 0.48))*octomap_free_cost; // Case 3: Free space, not traversable
  }

  // Copy data into cost_map_cloud
  int cost_map_size[3];
  for (int i=0; i<3; i++) cost_map_size[i] = round((cost_map_max[i] - cost_map_min[i])/voxel_size) + 1;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cost_map_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  for (int i=0; i<cost_map.size(); i++) {
    float query[3];
    index3_xyz(i, query, cost_map_min, cost_map_size, voxel_size);
    pcl::PointXYZI p;
    p.x = query[0];
    p.y = query[1];
    p.z = query[2];
    p.intensity = cost_map[i];
    cost_map_cloud->points.push_back(p);
  }

  // Convert cost_map_cloud to a PointCloud2 message and return
  sensor_msgs::PointCloud2 cost_map_msg;
  pcl::toROSMsg(*cost_map_cloud, cost_map_msg);
  cost_map_msg.header = cost_map_header;
  cost_map_msg.header.stamp = ros::Time::now();
  return cost_map_msg;
}


int main(int argc, char **argv)
{
  // Initialize ROS node with name and object instantiation
  ros::init(argc, argv, "costmap_generator");
  ros::NodeHandle n;

  // Declare subscribers and publishers
  ros::Subscriber sub_esdf = n.subscribe("esdf", 1, CallbackESDF);
  ros::Subscriber sub_octomap_binary = n.subscribe("octomap_binary", 1, callbackOctomapBinary);
  ros::Subscriber sub_octomap_full = n.subscribe("octomap_full", 1, callbackOctomapFull);
  ros::Publisher pub_cost_map = n.advertise<sensor_msgs::PointCloud2>("cost_map", 5);

  // ESDF value to use for untraversable voxels
  n.param("costmap_generator/octomap_free_cost", octomap_free_cost, (float)0.05);
  n.param("costmap_generator/unseen_cost", unseen_cost, (float)-1.0);

  // Declare and read in the node update rate from the launch file parameters
  double updateRate;
  n.param("costmap_generator/rate", updateRate, (double)10.0); // Hz
  ros::Rate r(updateRate);

  sensor_msgs::PointCloud2 fusedMsg;

  // Run the node until ROS quits
  while (ros::ok())
  {
    r.sleep(); // Node sleeps to update at a rate as close as possible to the updateRate parameter
    ros::spinOnce(); // All subscriber callbacks are called here.
    if (updated_occupancy && updated_ESDF) fusedMsg = FuseMaps();
    if (updated_occupancy_first && updated_ESDF_first) pub_cost_map.publish(fusedMsg);
  }

  return 0;
}
