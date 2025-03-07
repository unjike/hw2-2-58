#include "common.h"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>

// Apply the force from neighbor to particle
// static void inline __attribute__((always_inline))
static void inline apply_force(particle_t& particle, particle_t& particle_neigh) {
    // Calculate Distance
    double dx = particle_neigh.x - particle.x;
    double dy = particle_neigh.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = std::max(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
    particle_neigh.ay -= coef * dy;
    particle_neigh.ax -= coef * dx;
}

// Integrate the ODE
// static void inline __attribute__((always_inline))
static void inline move(particle_t& particle, double size) {
    // simplified Velocity integration
    // Energy conservation is better than explicit Euler method
    particle.vx += particle.ax * dt;
    particle.vy += particle.ay * dt;
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;

    // Bounce from walls
    while (particle.x < 0 || particle.x > size) {
        particle.x = particle.x < 0 ? -particle.x : 2 * size - particle.x;
        particle.vx = -particle.vx;
    }

    while (particle.y < 0 || particle.y > size) {
        particle.y = particle.y < 0 ? -particle.y : 2 * size - particle.y;
        particle.vy = -particle.vy;
    }
}

//Initializing Global Variables
int global_dims[2], global_coords[2], global_xd, global_yd, global_x0, global_y0, global_lda;
MPI_Comm global_comm;
const double grid_step = cutoff*1.00001;

//initializing custom data structures
enum ParticleKind {
    Owned,
    Ghost
};

enum Dimension {
    X_Dim = 0,
    Y_Dim = 1,
    Dims = 2
};

enum Direction {
    Down = 0,
    Up = 1,
    Dirs = 2
};

struct ParticleNode;
struct ParticleNodePredecessor;
struct ParticleNodePredecessor {
    ParticleNode* next = nullptr;
};

struct ParticleNode : ParticleNodePredecessor {
    ParticleNodePredecessor* prev = nullptr;
    particle_t* particle = nullptr;
    int global_i;
};

struct Neighbor {
    particle_t* send = nullptr;
    particle_t* recv = nullptr;
    int recvcnt = 0;
    int sendcnt = 0;
    int rank;
    
    MPI_Request send_req, recv_req;
    MPI_Status send_status, recv_status;


    void push(particle_t& particle) {
            send[sendcnt++] = particle;
    }
    void begin_send() {
        MPI_Issend(send, sendcnt, PARTICLE, rank, 0, global_comm, &send_req);
    }
    void finish_send() {
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        sendcnt = 0;
    }
    void receive() {
        MPI_Probe(rank, MPI_ANY_TAG, global_comm, &recv_status);
        MPI_Get_count(&recv_status, PARTICLE, &recvcnt);
        MPI_Recv(recv, recvcnt, PARTICLE, rank, MPI_ANY_TAG, global_comm, &recv_status);
    }
};

//initializing particles and nodes in 2D grid
Neighbor neighbors[Dims][Dirs];
ParticleNodePredecessor* particle_grid;
ParticleNode* particle_containers;
std::unordered_set<int> global_parts, global_ghosts;


// static void inline __attribute__((always_inline))
static void inline apply_intercell_force(particle_t* const particle, const int x_prime, const int y_prime) {
    int i_prime = x_prime+(global_xd+4)*y_prime;
    for(ParticleNode* pc_prime = particle_grid[i_prime].next; pc_prime != nullptr; pc_prime = pc_prime->next) {
        particle_t* particle_neigh = pc_prime->particle;
        apply_force(*particle, *particle_neigh);
    }
}




// static void inline __attribute__((always_inline))
static void inline linkParticle(ParticleNodePredecessor* pred, ParticleNode* pc) {
    pc->prev = pred;
    pc->next = pred->next;
    if (pred->next != nullptr)
        pred->next->prev = pc;
    pred->next = pc;
}

// static void inline __attribute__((always_inline))
static void inline unlinkParticle(ParticleNode* pc) {
    pc->prev->next = pc->next;
    if (pc->next != nullptr)
        pc->next->prev = pc->prev;
}



void init_simulation(particle_t* parts, int n_parts, double size, int rank, int num_procs) 
{
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here
    MPI_Dims_create(num_procs, Dims, global_dims);
    int periods[2] = {false, false};
    MPI_Cart_create(MPI_COMM_WORLD, Dims, global_dims, periods, false, &global_comm);
    MPI_Cart_coords(global_comm, rank, Dims, global_coords);
    for (int dim = 0; dim < Dims; ++dim) {
      MPI_Cart_shift(global_comm, dim, 1, &neighbors[dim][0].rank, &neighbors[dim][1].rank);
      for (int dir = 0; dir < Dirs; ++dir) {
        neighbors[dim][dir].send = new particle_t[n_parts];
        neighbors[dim][dir].recv = new particle_t[n_parts];
      }
    }

    global_lda = static_cast<int>(size/grid_step)+1;
    global_xd = (global_lda+global_dims[0]-1)/global_dims[0];
    global_yd = (global_lda+global_dims[1]-1)/global_dims[1];
    global_x0 = global_coords[0]*global_xd;
    global_y0 = global_coords[1]*global_yd;

    particle_grid = new ParticleNodePredecessor[(global_xd+4)*(global_yd+4)]();
    particle_containers = new ParticleNode[n_parts]();

    for (int p_i = 0; p_i < n_parts; ++p_i) {
        particle_t* particle = parts+p_i;
        int global_x = static_cast<int>(particle->x / grid_step)-global_x0;
        int global_y = static_cast<int>(particle->y / grid_step)-global_y0;
        ParticleNode* pc = particle_containers+p_i;
        
        particle->ax = particle->ay = 0;
        
        if (0 <= global_x && global_x < global_xd && 0 <= global_y && global_y < global_yd) {
            global_parts.insert(p_i);
            int global_i = global_x+2+(global_xd+4)*(global_y+2);
            pc->particle= particle;
            pc->global_i = global_i;
            ParticleNodePredecessor* g = particle_grid+global_i;
            linkParticle(g, pc);
        }
    }
}

// static void inline __attribute__((always_inline))
static void inline push_ghosts(
    Dimension dim, Direction dir)
{
    Neighbor& outgoing = neighbors[dim][dir];
    if (outgoing.rank != MPI_PROC_NULL) {
        int x_lo = dim == X_Dim && dir == Up ? global_xd-1 : 0;
        int x_hi = dim == X_Dim && dir == Down ? 0 : global_xd-1;
        int y_lo = dim == Y_Dim && dir == Up ? global_yd-1 : 0;
        int y_hi = dim == Y_Dim && dir == Down ? 0 : global_yd-1;
        for (int gy = 2+y_lo; gy <= 2+y_hi; ++gy) {
          for (int gx = 2+x_lo; gx <= 2+x_hi; ++gx) {
            int i = gx+(global_xd+4)*gy;
            for (ParticleNode* pc = particle_grid[i].next; pc != nullptr; pc = pc->next) {
                particle_t* particle = pc->particle;
                outgoing.push(*particle);
            }
          }
        }
    }
}

// static void inline __attribute__((always_inline))
static void inline neighbor_exchange(
    Dimension dim)
{
    Neighbor& neighbor_down = neighbors[dim][Down];
    Neighbor& neighbor_up = neighbors[dim][Up];
    neighbor_down.begin_send();
    neighbor_up.receive();
    neighbor_down.finish_send();
    neighbor_up.begin_send();
    neighbor_down.receive();
    neighbor_up.finish_send();
}

// static void inline __attribute__((always_inline))
static void inline receive_particles(
    particle_t* parts, ParticleNode* particle_containers, ParticleNodePredecessor* particle_grid,
    Dimension dim, Direction dir,
    bool pass_y, ParticleKind kind)
{
    Neighbor& incoming = neighbors[dim][dir];
    if (incoming.rank == MPI_PROC_NULL)
        return;
    for (particle_t* particle_recv = incoming.recv; particle_recv != incoming.recv+incoming.recvcnt; ++particle_recv) {
        int p_i = particle_recv->id - 1;
        ParticleNode* pc = particle_containers+p_i;
        particle_t* particle = parts+p_i;
        int global_x = static_cast<int>(particle_recv->x / grid_step)-global_x0;
        int global_y = static_cast<int>(particle_recv->y / grid_step)-global_y0;
        particle_recv->ax = particle_recv->ay = 0;
        if (pass_y) {
            if (kind != Ghost) {
              if (global_y < 0) {
                neighbors[Y_Dim][Down].push(*particle_recv);
                continue;
              } else if (global_y >= global_yd) {
                neighbors[Y_Dim][Up].push(*particle_recv);
                continue;
              }
            } else {
              if (global_y <= 0) {
                neighbors[Y_Dim][Down].push(*particle_recv);
              } else if (global_y >= global_yd-1) {
                neighbors[Y_Dim][Up].push(*particle_recv);
              }
            }
        }
        if (-1 <= global_x && global_x < global_xd+1 && -1 <= global_y && global_y < global_yd+1) {
            if (0 <= global_x && global_x < global_xd && 0 <= global_y && global_y < global_yd) {
                global_parts.insert(p_i);
            } else if (kind == Ghost) {
                global_ghosts.insert(p_i);
            }
            *particle = *particle_recv;
            pc->particle= particle;
            int global_i = global_x+2+(global_xd+4)*(global_y+2);
            pc->global_i = global_i;
            ParticleNodePredecessor* g = particle_grid+global_i;
            linkParticle(g, pc);
        }
    }
    incoming.recvcnt = 0;
}

void simulate_one_step(particle_t* parts, int n_parts, double size, int rank, int num_procs) 
{
    // Ghost particles interactions.
    // Along dimension 0.
    push_ghosts(X_Dim, Down);
    push_ghosts(X_Dim, Up);
    neighbor_exchange(X_Dim);
    receive_particles(parts, particle_containers, particle_grid, X_Dim, Down, true, Ghost);
    receive_particles(parts, particle_containers, particle_grid, X_Dim, Up, true, Ghost);
    // Along dimension 1, including particles from previous.
    push_ghosts(Y_Dim, Down);
    push_ghosts(Y_Dim, Up);
    neighbor_exchange(Y_Dim);
    receive_particles(parts, particle_containers, particle_grid, Y_Dim, Down, false, Ghost);
    receive_particles(parts, particle_containers, particle_grid, Y_Dim, Up, false, Ghost);
    // Apply Forces
    for (int gy = 1; gy < global_yd+3; ++gy) {
      for (int gx = 1; gx < global_xd+3; ++gx) {
        int i = gx+(global_xd+4)*gy;
        for (ParticleNode* pc = particle_grid[i].next; pc != nullptr; pc = pc->next) {
            particle_t* particle = pc->particle;
            apply_intercell_force(particle, gx-1, gy-1);
            apply_intercell_force(particle, gx, gy-1);
            apply_intercell_force(particle, gx+1, gy-1);
            apply_intercell_force(particle, gx-1, gy);
            for (ParticleNode* pc_prime = pc->next; pc_prime != nullptr; pc_prime = pc_prime->next) apply_force(*particle, *pc_prime->particle);
        }
      }
    }

    // Move Particles
    for (auto it = global_parts.begin(); it != global_parts.end(); ) {
        int p_i = *it;
        ParticleNode* pc = particle_containers+p_i;
        particle_t* particle = parts+p_i;
        move(*particle, size);
            particle->ax = particle->ay = 0;
        int global_x_prime = static_cast<int>(particle->x / grid_step)-global_x0;
        int global_y_prime = static_cast<int>(particle->y / grid_step)-global_y0;
        int global_i_prime = global_x_prime+2+(global_xd+4)*(global_y_prime+2);
        if (pc->global_i != global_i_prime) {
            if (global_x_prime < 0 || global_x_prime >= global_xd || global_y_prime < 0 || global_y_prime >= global_yd) {
              unlinkParticle(pc);
              it = global_parts.erase(it);
              if (global_x_prime < 0) {
                neighbors[X_Dim][Down].push(*particle);
              } else if (global_x_prime >= global_xd) {
                neighbors[X_Dim][Up].push(*particle);
              } else if (global_y_prime < 0) {
                neighbors[Y_Dim][Down].push(*particle);
              } else if (global_y_prime >= global_yd) {
                neighbors[Y_Dim][Up].push(*particle);
              }
              continue;
            }
            ParticleNodePredecessor* global_prime = particle_grid+global_i_prime;
            pc->global_i = global_i_prime;
            unlinkParticle(pc);
            linkParticle(global_prime, pc);
        }
        ++it;
    }

    for (const int p_i: global_ghosts) {
        unlinkParticle(particle_containers+p_i);
    }

    global_ghosts.clear();
    // Move particles.
    // interaction along dim 0.
    neighbor_exchange(X_Dim);
    receive_particles(parts, particle_containers, particle_grid, X_Dim, Down, true, Owned);
    receive_particles(parts, particle_containers, particle_grid, X_Dim, Up, true, Owned);
    // interaction along dim 1
    neighbor_exchange(Y_Dim);
    receive_particles(parts, particle_containers, particle_grid, Y_Dim, Down, false, Owned);
    receive_particles(parts, particle_containers, particle_grid, Y_Dim, Up, false, Owned);
}

void gather_for_save(particle_t* parts, int n_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    MPI_Barrier(global_comm);
    std::vector<particle_t> send;
    send.reserve(global_parts.size());
    std::vector<particle_t> recv;
    for (const int p_i: global_parts) {
        send.push_back(parts[p_i]);
    }
    int sendcount = send.size();
    int* recvcounts, *recvdispls;
    if (rank == 0) {
        recvcounts = new int[num_procs];
        recvdispls = new int[num_procs];
    }
    MPI_Barrier(global_comm);
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, global_comm);
    if (rank == 0) {
        recvdispls[0] = 0;
        for (int i = 1; i < num_procs; ++i) recvdispls[i] = recvdispls[i-1]+recvcounts[i-1];
        recv.resize(recvdispls[num_procs-1]+recvcounts[num_procs-1]);
    }
    MPI_Barrier(global_comm);
    MPI_Gatherv(send.data(), sendcount, PARTICLE, recv.data(), recvcounts, recvdispls, PARTICLE, 0, global_comm);
    if (rank == 0) {
        for (const auto& particle_recv: recv) {
            int p_i = particle_recv.id - 1;
            parts[p_i] = particle_recv;
        }
    }
}
