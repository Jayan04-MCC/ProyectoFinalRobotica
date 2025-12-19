#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <webots/distance_sensor.h>
#include <webots/led.h>
#include <webots/light_sensor.h>
#include <webots/motor.h>
#include <webots/robot.h>

// Global defines
#define TRUE 1
#define FALSE 0
#define NO_SIDE -1
#define LEFT 0
#define RIGHT 1
#define WHITE 0
#define BLACK 1
#define SIMULATION 0  // for wb_robot_get_mode() function
#define REALITY 2     // for wb_robot_get_mode() function
#define TIME_STEP 32  // [ms]

// 8 IR proximity sensors
#define NB_DIST_SENS 8
#define PS_RIGHT_00 0
#define PS_RIGHT_45 1
#define PS_RIGHT_90 2
#define PS_RIGHT_REAR 3
#define PS_LEFT_REAR 4
#define PS_LEFT_90 5
#define PS_LEFT_45 6
#define PS_LEFT_00 7
WbDeviceTag ps[NB_DIST_SENS]; /* proximity sensors */
int ps_value[NB_DIST_SENS] = {0, 0, 0, 0, 0, 0, 0, 0};
const int PS_OFFSET_SIMULATION[NB_DIST_SENS] = {300, 300, 300, 300, 300, 300, 300, 300};
// *** TO BE ADAPTED TO YOUR ROBOT ***
const int PS_OFFSET_REALITY[NB_DIST_SENS] = {480, 170, 320, 500, 600, 680, 210, 640};

// 3 IR ground color sensors
#define NB_GROUND_SENS 3
#define GS_WHITE 900
#define GS_LEFT 0
#define GS_CENTER 1
#define GS_RIGHT 2
WbDeviceTag gs[NB_GROUND_SENS]; /* ground sensors */
unsigned short gs_value[NB_GROUND_SENS] = {0, 0, 0};

// Motors
WbDeviceTag left_motor, right_motor;

// LEDs
#define NB_LEDS 8
WbDeviceTag led[NB_LEDS];

//------------------------------------------------------------------------------
//
//    BEHAVIORAL MODULES
//
//------------------------------------------------------------------------------

////////////////////////////////////////////
// LFM - Line Following Module
//
// This module implements a very simple, Braitenberg-like behavior in order
// to follow a black line on the ground. Output speeds are stored in
// lfm_speed[LEFT] and lfm_speed[RIGHT].

int lfm_speed[2];

#define LFM_FORWARD_SPEED 500
#define LFM_K_GS_SPEED 0.4

#define KP 1.7
#define KI 0.001
#define KD 0.0001
#define MAX_CORRECTION 450.0
#define MAX_SPEED 550.0

#define MAX_GS_VALUE 1000.0
#define MAX_WHEEL_SPEED 550.0

void pid(void) {
  static double prev_error = 0.0;
  static double integral = 0.0;
  double error, derivative, correction;
  double dt = (double)TIME_STEP / 1000.0;

  error = gs_value[GS_RIGHT] - gs_value[GS_LEFT];

  integral += error * dt;
  derivative = (error - prev_error) / dt;
  correction = KP * error + KI * integral + KD * derivative;

  if (correction > MAX_CORRECTION) correction = MAX_CORRECTION;
  if (correction < -MAX_CORRECTION) correction = -MAX_CORRECTION;

  lfm_speed[LEFT]  = LFM_FORWARD_SPEED - correction;
  lfm_speed[RIGHT] = LFM_FORWARD_SPEED + correction;

  if (lfm_speed[LEFT] > MAX_SPEED) lfm_speed[LEFT] = MAX_SPEED;
  if (lfm_speed[LEFT] < -MAX_SPEED) lfm_speed[LEFT] = -MAX_SPEED;
  if (lfm_speed[RIGHT] > MAX_SPEED) lfm_speed[RIGHT] = MAX_SPEED;
  if (lfm_speed[RIGHT] < -MAX_SPEED) lfm_speed[RIGHT] = -MAX_SPEED;

  printf("prevError=%.3f, error=%.3f, integral=%.3f, derivative=%.3f, correction=%.3f\n", prev_error, error, integral, derivative, correction);

  prev_error = error;
}

void set_ddqn_state(double state[6],
                    unsigned short gs_value[3],
                    int speed_left,
                    int speed_right) {

  // 1. Sensores de suelo (normalizados)
  state[0] = gs_value[GS_LEFT];
  state[1] = gs_value[GS_CENTER];
  state[2] = gs_value[GS_RIGHT];

  // 3. Velocidades de las ruedas (normalizadas)
  state[3] = speed_left;
  state[4] = speed_right;
}

void apply_ddqn_action(int action, int speed[2]) {
  switch (action) {
    case 0:  // girar izquierda
      speed[LEFT]  = 300;
      speed[RIGHT] = 550;
      break;

    case 1:  // avanzar recto
      speed[LEFT]  = 500;
      speed[RIGHT] = 500;
      break;

    case 2:  // girar derecha
      speed[LEFT]  = 550;
      speed[RIGHT] = 300;
      break;

    default:
      speed[LEFT]  = 0;
      speed[RIGHT] = 0;
      break;
  }

  if (speed[LEFT] > MAX_SPEED) speed[LEFT] = MAX_SPEED;
  if (speed[LEFT] < -MAX_SPEED) speed[LEFT] = -MAX_SPEED;
  if (speed[RIGHT] > MAX_SPEED) speed[RIGHT] = MAX_SPEED;
  if (speed[RIGHT] < -MAX_SPEED) speed[RIGHT] = -MAX_SPEED;
}

////////////////////////////////////////////
// OAM - Obstacle Avoidance Module
//
// The OAM routine first detects obstacles in front of the robot, then records
// their side in "oam_side" and avoid the detected obstacle by
// turning away according to very simple weighted connections between
// proximity sensors and motors. "oam_active" becomes active when as soon as
// an object is detected and "oam_reset" inactivates the module and set
// "oam_side" to NO_SIDE. Output speeds are in oam_speed[LEFT] and oam_speed[RIGHT].

int oam_active, oam_reset;
int oam_speed[2];
int oam_side = NO_SIDE;

#define OAM_OBST_THRESHOLD 100
#define OAM_FORWARD_SPEED 150
#define OAM_K_PS_90 0.2
#define OAM_K_PS_45 0.9
#define OAM_K_PS_00 1.2
#define OAM_K_MAX_DELTAS 600

void ObstacleAvoidanceModule(void) {
  int max_ds_value, i;
  int Activation[] = {0, 0};

  // Module RESET
  if (oam_reset) {
    oam_active = FALSE;
    oam_side = NO_SIDE;
  }
  oam_reset = 0;

  // Determine the presence and the side of an obstacle
  max_ds_value = 0;
  for (i = PS_RIGHT_00; i <= PS_RIGHT_45; i++) {
    if (max_ds_value < ps_value[i])
      max_ds_value = ps_value[i];
    Activation[RIGHT] += ps_value[i];
  }
  for (i = PS_LEFT_45; i <= PS_LEFT_00; i++) {
    if (max_ds_value < ps_value[i])
      max_ds_value = ps_value[i];
    Activation[LEFT] += ps_value[i];
  }
  if (max_ds_value > OAM_OBST_THRESHOLD)
    oam_active = TRUE;

  if (oam_active && oam_side == NO_SIDE)  // check for side of obstacle only when not already detected
  {
    if (Activation[RIGHT] > Activation[LEFT])
      oam_side = RIGHT;
    else
      oam_side = LEFT;
  }

  // Forward speed
  oam_speed[LEFT] = OAM_FORWARD_SPEED;
  oam_speed[RIGHT] = OAM_FORWARD_SPEED;

  // Go away from obstacle
  if (oam_active) {
    int DeltaS = 0;
    // The rotation of the robot is determined by the location and the side of the obstacle
    if (oam_side == LEFT) {
      //(((ps_value[PS_LEFT_90]-PS_OFFSET)<0)?0:(ps_value[PS_LEFT_90]-PS_OFFSET)));
      DeltaS -= (int)(OAM_K_PS_90 * ps_value[PS_LEFT_90]);
      //(((ps_value[PS_LEFT_45]-PS_OFFSET)<0)?0:(ps_value[PS_LEFT_45]-PS_OFFSET)));
      DeltaS -= (int)(OAM_K_PS_45 * ps_value[PS_LEFT_45]);
      //(((ps_value[PS_LEFT_00]-PS_OFFSET)<0)?0:(ps_value[PS_LEFT_00]-PS_OFFSET)));
      DeltaS -= (int)(OAM_K_PS_00 * ps_value[PS_LEFT_00]);
    } else {  // oam_side == RIGHT
      //(((ps_value[PS_RIGHT_90]-PS_OFFSET)<0)?0:(ps_value[PS_RIGHT_90]-PS_OFFSET)));
      DeltaS += (int)(OAM_K_PS_90 * ps_value[PS_RIGHT_90]);
      //(((ps_value[PS_RIGHT_45]-PS_OFFSET)<0)?0:(ps_value[PS_RIGHT_45]-PS_OFFSET)));
      DeltaS += (int)(OAM_K_PS_45 * ps_value[PS_RIGHT_45]);
      //(((ps_value[PS_RIGHT_00]-PS_OFFSET)<0)?0:(ps_value[PS_RIGHT_00]-PS_OFFSET)));
      DeltaS += (int)(OAM_K_PS_00 * ps_value[PS_RIGHT_00]);
    }
    if (DeltaS > OAM_K_MAX_DELTAS)
      DeltaS = OAM_K_MAX_DELTAS;
    if (DeltaS < -OAM_K_MAX_DELTAS)
      DeltaS = -OAM_K_MAX_DELTAS;

    // Set speeds
    oam_speed[LEFT] -= DeltaS;
    oam_speed[RIGHT] += DeltaS;
  }
}

////////////////////////////////////////////
// LLM - Line Leaving Module
//
// Since it has no output, this routine is not completely finished. It has
// been designed to monitor the moment while the robot is leaving the
// track and signal to other modules some related events. It becomes active
// whenever the "side" variable displays a rising edge (changing from -1 to 0 or 1).

int llm_active = FALSE, llm_inibit_ofm_speed, llm_past_side = NO_SIDE;

#define LLM_THRESHOLD 800

void LineLeavingModule(int side) {
  // Starting the module on a rising edge of "side"
  if (!llm_active && side != NO_SIDE && llm_past_side == NO_SIDE)
    llm_active = TRUE;

  // Updating the memory of the "side" state at the previous call
  llm_past_side = side;

  // Main loop
  if (llm_active)  // Simply waiting until the line is not detected anymore
  {
    if (side == LEFT) {
      if ((gs_value[GS_CENTER] + gs_value[GS_LEFT]) / 2 > LLM_THRESHOLD)  // out of line
      {
        llm_active = FALSE;
        // *** PUT YOUR CODE HERE ***
      } else  // still leaving the line
      {
        // *** PUT YOUR CODE HERE ***
      }
    } else  // side == RIGHT
    {
      if ((gs_value[GS_CENTER] + gs_value[GS_RIGHT]) / 2 > LLM_THRESHOLD)  // out of line
      {
        llm_active = FALSE;
        // *** PUT YOUR CODE HERE ***
      } else  // still leaving the line
      {
        // *** PUT YOUR CODE HERE ***
      }
    }
  }
}

////////////////////////////////////////////
// OFM - Obstacle Following Module
//
// This function just gives the robot a tendency to steer toward the side
// indicated by its argument "side". When used in competition with OAM it
// gives rise to an object following behavior. The output speeds are
// stored in ofm_speed[LEFT] and ofm_speed[RIGHT].

int ofm_active;
int ofm_speed[2];

#define OFM_DELTA_SPEED 150

void ObstacleFollowingModule(int side) {
  if (side != NO_SIDE) {
    ofm_active = TRUE;
    if (side == LEFT) {
      ofm_speed[LEFT] = -OFM_DELTA_SPEED;
      ofm_speed[RIGHT] = OFM_DELTA_SPEED;
    } else {
      ofm_speed[LEFT] = OFM_DELTA_SPEED;
      ofm_speed[RIGHT] = -OFM_DELTA_SPEED;
    }
  } else  // side = NO_SIDE
  {
    ofm_active = FALSE;
    ofm_speed[LEFT] = 0;
    ofm_speed[RIGHT] = 0;
  }
}

////////////////////////////////////////////
// LEM - Line Entering Module
//
// This is the most complex module (and you might find easier to re-program it
// by yourself instead of trying to understand it ;-). Its purpose is to handle
// the moment when the robot must re-enter the track (after having by-passed
// an obstacle, e.g.). It is organized like a state machine, which state is
// stored in "lem_state" (see LEM_STATE_STANDBY and following #defines).
// The inputs are (i) the two lateral ground sensors, (ii) the argument "side"
// which determines the direction that the robot has to follow when detecting
// a black line, and (iii) the variable "lem_reset" that resets the state to
// standby. The output speeds are stored in lem_speed[LEFT] and
// lem_speed[RIGHT].

int lem_active;
int lem_reset;
int lem_speed[2];
int lem_state, lem_black_counter;
int cur_op_gs_value, prev_op_gs_value;

#define LEM_FORWARD_SPEED 100
#define LEM_K_GS_SPEED 0.8
#define LEM_THRESHOLD 500

#define LEM_STATE_STANDBY 0
#define LEM_STATE_LOOKING_FOR_LINE 1
#define LEM_STATE_LINE_DETECTED 2
#define LEM_STATE_ON_LINE 3

void LineEnteringModule(int side) {
  int Side, OpSide, GS_Side, GS_OpSide;

  // Module reset
  if (lem_reset)
    lem_state = LEM_STATE_LOOKING_FOR_LINE;
  lem_reset = FALSE;

  // Initialization
  lem_speed[LEFT] = LEM_FORWARD_SPEED;
  lem_speed[RIGHT] = LEM_FORWARD_SPEED;
  if (side == LEFT) {  // if obstacle on left side -> enter line rightward
    Side = RIGHT;      // line entering direction
    OpSide = LEFT;
    GS_Side = GS_RIGHT;
    GS_OpSide = GS_LEFT;
  } else {        // if obstacle on left side -> enter line leftward
    Side = LEFT;  // line entering direction
    OpSide = RIGHT;
    GS_Side = GS_LEFT;
    GS_OpSide = GS_RIGHT;
  }

  // Main loop (state machine)
  switch (lem_state) {
    case LEM_STATE_STANDBY:
      lem_active = FALSE;
      break;
    case LEM_STATE_LOOKING_FOR_LINE:
      if (gs_value[GS_Side] < LEM_THRESHOLD) {
        lem_active = TRUE;
        // set speeds for entering line
        lem_speed[OpSide] = LEM_FORWARD_SPEED;
        lem_speed[Side] = LEM_FORWARD_SPEED;  // - LEM_K_GS_SPEED * gs_value[GS_Side];
        lem_state = LEM_STATE_LINE_DETECTED;
        // save ground sensor value
        if (gs_value[GS_OpSide] < LEM_THRESHOLD) {
          cur_op_gs_value = BLACK;
          lem_black_counter = 1;
        } else {
          cur_op_gs_value = WHITE;
          lem_black_counter = 0;
        }
        prev_op_gs_value = cur_op_gs_value;
      }
      break;
    case LEM_STATE_LINE_DETECTED:
      // save the oposite ground sensor value
      if (gs_value[GS_OpSide] < LEM_THRESHOLD) {
        cur_op_gs_value = BLACK;
        lem_black_counter++;
      } else
        cur_op_gs_value = WHITE;
      // detect the falling edge BLACK->WHITE
      if (prev_op_gs_value == BLACK && cur_op_gs_value == WHITE) {
        lem_state = LEM_STATE_ON_LINE;
        lem_speed[OpSide] = 0;
        lem_speed[Side] = 0;
      } else {
        prev_op_gs_value = cur_op_gs_value;
        // set speeds for entering line
        lem_speed[OpSide] = LEM_FORWARD_SPEED + LEM_K_GS_SPEED * (GS_WHITE - gs_value[GS_Side]);
        lem_speed[Side] = LEM_FORWARD_SPEED - LEM_K_GS_SPEED * (GS_WHITE - gs_value[GS_Side]);
      }
      break;
    case LEM_STATE_ON_LINE:
      oam_reset = TRUE;
      lem_active = FALSE;
      lem_state = LEM_STATE_STANDBY;
      break;
  }
}

//------------------------------------------------------------------------------
//
//    CONTROLLER
//
//------------------------------------------------------------------------------

int sockfd;
struct sockaddr_in server_addr;





////////////////////////////////////////////
// Main
int main() {
  int i, speed[2], ps_offset[NB_DIST_SENS] = {0, 0, 0, 0, 0, 0, 0, 0}, Mode = 1;

  /* intialize Webots */
  wb_robot_init();

  // ================================================================
  // CONEXIÓN AL TRAINER (SERVIDOR DDQN EN JETSON)
  // ================================================================
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    perror("socket");
    return 1;
  }

  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(5000);
  inet_pton(AF_INET, "10.7.50.140", &server_addr.sin_addr); // IP del TRAINER

  if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    perror("connect");
    close(sockfd);
    return 1;
  }

  printf("[ENV] Conectado al DDQN Trainer\n");

  // ================================================================
  // INICIALIZACIÓN DE DISPOSITIVOS WEBOTS
  // ================================================================
  char name[20];
  
  // LEDs
  for (i = 0; i < NB_LEDS; i++) {
    sprintf(name, "led%d", i);
    led[i] = wb_robot_get_device(name);
  }
  
  // Sensores de proximidad
  for (i = 0; i < NB_DIST_SENS; i++) {
    sprintf(name, "ps%d", i);
    ps[i] = wb_robot_get_device(name);
    wb_distance_sensor_enable(ps[i], TIME_STEP);
  }
  
  // Sensores de suelo
  for (i = 0; i < NB_GROUND_SENS; i++) {
    sprintf(name, "gs%d", i);
    gs[i] = wb_robot_get_device(name);
    wb_distance_sensor_enable(gs[i], TIME_STEP);
  }
  
  // Motores
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);

  // Inicializar velocidades
  speed[LEFT] = 0;
  speed[RIGHT] = 0;

  // ================================================================
  // LOOP PRINCIPAL - SINCRONIZADO CON TRAINER
  // ================================================================
  for (;;) {
    // ────────────────────────────────────────────────────────────
    // 1. EJECUTAR UN PASO DE SIMULACIÓN
    // ────────────────────────────────────────────────────────────
    if (wb_robot_step(TIME_STEP) == -1) {
      break;  // Simulación terminada
    }

    // ────────────────────────────────────────────────────────────
    // 2. MANEJAR CAMBIO DE MODO (SIMULACIÓN <-> REALIDAD)
    // ────────────────────────────────────────────────────────────
    if (Mode != wb_robot_get_mode()) {
      oam_reset = TRUE;
      llm_active = FALSE;
      llm_past_side = NO_SIDE;
      ofm_active = FALSE;
      lem_active = FALSE;
      lem_state = LEM_STATE_STANDBY;
      Mode = wb_robot_get_mode();
      
      if (Mode == SIMULATION) {
        for (i = 0; i < NB_DIST_SENS; i++)
          ps_offset[i] = PS_OFFSET_SIMULATION[i];
        wb_motor_set_velocity(left_motor, 0);
        wb_motor_set_velocity(right_motor, 0);
        wb_robot_step(TIME_STEP);
        printf("\n\n\nSwitching to SIMULATION and reseting all BB variables.\n\n");
      } else if (Mode == REALITY) {
        for (i = 0; i < NB_DIST_SENS; i++)
          ps_offset[i] = PS_OFFSET_REALITY[i];
        wb_motor_set_velocity(left_motor, 0);
        wb_motor_set_velocity(right_motor, 0);
        wb_robot_step(TIME_STEP);
        printf("\n\n\nSwitching to REALITY and reseting all BB variables.\n\n");
      }
    }

    // ────────────────────────────────────────────────────────────
    // 3. LEER VALORES DE SENSORES
    // ────────────────────────────────────────────────────────────
    for (i = 0; i < NB_DIST_SENS; i++) {
      ps_value[i] = (((int)wb_distance_sensor_get_value(ps[i]) - ps_offset[i]) < 0) ?
                      0 :
                      ((int)wb_distance_sensor_get_value(ps[i]) - ps_offset[i]);
    }
    
    for (i = 0; i < NB_GROUND_SENS; i++) {
      gs_value[i] = wb_distance_sensor_get_value(gs[i]);
    }

    // ────────────────────────────────────────────────────────────
    // 4. PREPARAR ESTADO PARA ENVIAR AL TRAINER
    // ────────────────────────────────────────────────────────────
    int32_t data[5];
    data[0] = (int32_t)gs_value[GS_LEFT];
    data[1] = (int32_t)gs_value[GS_CENTER];
    data[2] = (int32_t)gs_value[GS_RIGHT];
    data[3] = (int32_t)speed[LEFT];    // Velocidad izquierda
    data[4] = (int32_t)speed[RIGHT];   // Velocidad derecha

    // ────────────────────────────────────────────────────────────
    // 5. ENVIAR TAMAÑO DE DATOS
    // ────────────────────────────────────────────────────────────
    uint32_t data_size = htonl(5);  // 5 elementos en el array
    ssize_t sent = send(sockfd, &data_size, sizeof(data_size), 0);
    if (sent <= 0) {
      printf("[ENV] Error enviando data_size o conexión cerrada\n");
      break;
    }

    // ────────────────────────────────────────────────────────────
    // 6. ENVIAR DATOS DEL ESTADO
    // ────────────────────────────────────────────────────────────
    sent = send(sockfd, data, sizeof(data), 0);
    if (sent <= 0) {
      printf("[ENV] Error enviando data o conexión cerrada\n");
      break;
    }

    // ────────────────────────────────────────────────────────────
    // 7. RECIBIR ACCIÓN DEL TRAINER
    // ────────────────────────────────────────────────────────────
    int32_t net_action;
    ssize_t received = recv(sockfd, &net_action, sizeof(net_action), 0);
    if (received <= 0) {
      printf("[ENV] Error recibiendo action o conexión cerrada\n");
      break;
    }
    
    int action = ntohl(net_action);

    // ────────────────────────────────────────────────────────────
    // 8. APLICAR ACCIÓN RECIBIDA
    // ────────────────────────────────────────────────────────────
    apply_ddqn_action(action, speed);

    // ────────────────────────────────────────────────────────────
    // 9. EJECUTAR ACCIÓN EN LOS MOTORES
    // ────────────────────────────────────────────────────────────
    wb_motor_set_velocity(left_motor, 0.00628 * speed[LEFT]);
    wb_motor_set_velocity(right_motor, 0.00628 * speed[RIGHT]);

    // ────────────────────────────────────────────────────────────
    // NOTA: En el siguiente ciclo del loop:
    // - wb_robot_step() ejecutará la acción durante TIME_STEP ms
    // - Los sensores leerán el NUEVO estado resultante
    // - Enviaremos ese nuevo estado al TRAINER
    // 
    // Este es el ciclo correcto de Reinforcement Learning:
    // Estado → Acción → Nuevo Estado → Reward → Repetir
    // ────────────────────────────────────────────────────────────
  }

  // ================================================================
  // LIMPIEZA Y CIERRE
  // ================================================================
  close(sockfd);
  printf("[ENV] Conexión cerrada, simulación terminada\n");
  return 0;
}
