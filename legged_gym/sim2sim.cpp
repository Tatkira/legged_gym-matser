// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Geometry>
#include <torch/torch.h>
#include <torch/script.h> // JIT支持
#include <chrono>         // 时间测量
#include <thread>         // 延迟功能


// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// auto device = torch::kCPU;
// struct obs_scales

auto device = torch::kCPU;

struct obs_scales
{
  float lin_vel = 2.0;
  float ang_vel = 0.25;
  float dof_pos = 1.0;
  float dof_vel = 0.05;
  float height_measurements = 5.0;
};

obs_scales scales;
float action_scale = 0.25;
std::vector<float> actions;
std::vector<float> default_joint_angles = {0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5};
std::vector<float> commands_scale = {2.0000, 2.0000, 0.2500};

const double measured_points_x[] = {-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                                  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
const int num_points_x = 17;

const double measured_points_y[] = {-0.5, -0.4, -0.3, -0.2, -0.1,
                                  0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
const int num_points_y = 11;

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
 (void)window;
    (void)scancode;
    (void)mods;

    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    (void)button;
    (void)act;
    (void)mods;

    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    (void)window;
    (void)xoffset;

    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

std::vector<mjtNum> get_sensor_data(const mjModel *model, const mjData *data, const std::string &sensor_name)
{
  int sensor_id = mj_name2id(model, mjOBJ_SENSOR, sensor_name.c_str());
  if (sensor_id == -1)
  {
    std::cout << "no found sensor" << std::endl;
    return std::vector<mjtNum>();
  }
  int data_pos = 0;
  for (int i = 0; i < sensor_id; i++)
  {
    data_pos += model->sensor_dim[i];
  }
  std::vector<mjtNum> sensor_data(model->sensor_dim[sensor_id]);
  for (size_t i = 0; i < sensor_data.size(); i++)
  {
    sensor_data[i] = data->sensordata[data_pos + i];
  }
  return sensor_data;
}

// v'=q^-1*v*q
std::vector<double> world2self(std::vector<double> &quat, std::vector<double> v)
{
  Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
  Eigen::Vector3d v_vec(v[0], v[1], v[2]);
  double q_w = q.w();
  Eigen::Vector3d q_vec = q.vec();
  Eigen::Vector3d a = v_vec * (2.0 * q_w * q_w - 1.0);
  Eigen::Vector3d b = q_vec.cross(v_vec) * q_w * 2.0;
  Eigen::Vector3d c = q_vec * (q_vec.dot(v_vec)) * 2.0;
  Eigen::Vector3d result = a - b + c;
  std::vector<double> world_angle_speed = {result.x(), result.y(), result.z()};
  return world_angle_speed;
}

std::vector<float> compute_ctrl(std::vector<float> act)
{
  // 缩放
  for (int i = 0; i < 12; i++)
  {
    act[i] = act[i] * action_scale + default_joint_angles[i];
  }
  return act;
}


// 高度检测函数
double getTerrainHeight(mjModel* model, mjData* data, double x, double y) {
    // 获取机器人基座位置
    int robot_body_id = mj_name2id(model, mjOBJ_BODY, "robot");
    double* base_pos = data->xpos + 3 * robot_body_id;
    double* base_quat = data->xquat + 4 * robot_body_id;

    // 将局部坐标转换为全局坐标
    double local_pos[3] = {x, y, 0};
    double global_pos[3];
    mju_rotVecQuat(global_pos, local_pos, base_quat);
    global_pos[0] += base_pos[0];
    global_pos[1] += base_pos[1];

    // 设置射线起点（高于地面足够高度）
    double ray_start[3] = {global_pos[0], global_pos[1], 10.0};
    double ray_dir[3] = {0, 0, -1}; // 向下射线

    // 排除机器人自身进行检测
    // 修复参数：geomgroup设为NULL，flg_static设为0，group设为0，geom_id改为int
    int geom_id;
    mjtNum distance = mj_ray(model, data, ray_start, ray_dir, NULL, 0, 0, &geom_id);


    if (distance >= 0) {
         double terrain_height = (ray_start[2] + distance * ray_dir[2]) - base_pos[2];
        // 添加调试输出
        static bool first = true;
        if(first) {
            std::cout << "Height measurement test:" << std::endl;
            std::cout << "Global pos: (" << global_pos[0] << ", " << global_pos[1] << ")" << std::endl;
            std::cout << "Detected height: " << terrain_height << std::endl;
            first = false;
        }
        return terrain_height;
    }
    return 0.0; // 未检测到时返回0
}

torch::Tensor compute_observations()
{
// 1. 获取基座数据
  auto base_quat = get_sensor_data(m, d, "orientation");
  std::vector<float> obs;
  auto base_lin_vel = get_sensor_data(m, d, "base_lin_vel");
  base_lin_vel = world2self(base_quat, base_lin_vel);
  for (size_t i = 0; i < base_lin_vel.size(); i++)
  {
    base_lin_vel[i] *= scales.lin_vel;
    obs.push_back(base_lin_vel[i]);
  }

  auto base_ang_vel = get_sensor_data(m, d, "base_ang_vel");
  base_ang_vel = world2self(base_quat, base_ang_vel);
  for (size_t i = 0; i < base_ang_vel.size(); i++)
  {
    base_ang_vel[i] *= scales.ang_vel;
    obs.push_back(base_ang_vel[i]);
  }
  // 并非加速度计
  std::vector<double> gravity_vec = {0.0, 0.0, -1.0};
  auto projected_gravity = world2self(base_quat, gravity_vec);
  for (auto &i : projected_gravity)
    obs.push_back(i);

  // command
  std::vector<float> commands = {0.7, 0.0, 0.0};
  for (size_t i = 0; i < commands.size(); i++)
  {
    commands[i] *= commands_scale[i];
    obs.push_back(commands[i]);
  }
  // dof_pos
  std::vector<float> dof_pos;
  for (int i = 0; i < 12; i++)
  {
    dof_pos.push_back((d->sensordata[i]));
    obs.push_back((d->sensordata[i] - default_joint_angles[i]) * scales.dof_pos);
  }
  // dof_vel
  std::vector<float> dof_vel;
  for (int i = 12; i < 24; i++)
  {
    dof_vel.push_back(d->sensordata[i]);
    obs.push_back(d->sensordata[i] * scales.dof_vel);
  }
  // actons是剪裁后的
  if (actions.size() < 12)
  {
    for (int i = 0; i < 12; i++)
    {
      actions.push_back(0);
    }
  }
  for (auto &i : actions) {
    obs.push_back(i);
}

     // 2. 添加地形高度测量（新增部分）
    for (int xi = 0; xi < num_points_x; xi++) {
        for (int yi = 0; yi < num_points_y; yi++) {
            double x = measured_points_x[xi];
            double y = measured_points_y[yi];
            double height = getTerrainHeight(m, d, x, y);
            // 应用高度缩放因子
            height *= scales.height_measurements;
            obs.push_back(height);
        }
    }

  std::cout << "Observation vector size: " << obs.size() << std::endl;
  std::cout << "First 10 elements: ";
    for(int i=0; i<10; i++) std::cout << obs[i] << " ";
    std::cout << std::endl;

  torch::Tensor out = torch::from_blob(obs.data(), {static_cast<long>(obs.size())}, torch::kFloat32);
  return out.to(device);
}





//============================================================================
// main function
int main(int argc, const char** argv)
{
    // check command-line arguments
    if( argc!=2 )
    {
        printf(" USAGE:  basic modelfile\n");
        return 0;
    }

    // 输出mujoco版本信息
    std::cout << "MuJoCo version: " << mj_version() << std::endl;

    // load and compile model
    char error[1000] = "Could not load binary model";
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[1], 0);
    else
        m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
 //---------------------------jit-------------------------------
    mj_step(m, d);
    compute_observations();
    // 注意：将路径修改为你的实际模型路径
    // torch::jit::script::Module module = torch::jit::load("../logs/a1_flat_policy.pt", device);
    torch::jit::script::Module module = torch::jit::load("/home/one/isaac/legged_gym-master/logs/rough_a1/exported/policies/policy_1.pt", device);


    while (!glfwWindowShouldClose(window))
    {
      auto obs = compute_observations();
      torch::jit::Stack inputs;
      inputs.push_back(obs.to(device));

      auto start = std::chrono::high_resolution_clock::now();
      // 获取输出
      auto o = module.forward(std::move(inputs));
      at::Tensor output = o.toTensor().cpu();
      // 裁减action
      output = torch::clip(output, -100, 100);
      std::vector<float> vec(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
      actions = vec;

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration_ms = end - start;
      std::cout << "Time taken: " << duration_ms.count() << " ms" << std::endl;

      auto a = compute_ctrl(actions);
      for (int i = 0; i < 12; i++)
      {
        d->ctrl[i] = a[i];
      }

      mj_step(m, d);
      mj_step(m, d);
      mj_step(m, d);
      mj_step(m, d);
      // 延迟方便查看仿真
      std::this_thread::sleep_for(std::chrono::milliseconds(30));

      // get framebuffer viewport
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
      // update scene and render
      mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
      mjr_render(viewport, &scn, &con);
      // swap OpenGL buffers (blocking call due to v-sync)
      glfwSwapBuffers(window);
      // process pending GUI events, call GLFW callbacks
      glfwPollEvents();
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}
