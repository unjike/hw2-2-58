// #include "common.h"
// #include <mpi.h>
// #include <cmath>
// #include <vector>
// #include <algorithm>

// // Apply the force from neighbor to particle
// void apply_force(particle_t& particle, particle_t& neighbor) {
//     // Calculate Distance
//     double dx = neighbor.x - particle.x;
//     double dy = neighbor.y - particle.y;
//     double r2 = dx * dx + dy * dy;

//     // Check if the two particles should interact
//     if (r2 > cutoff * cutoff)
//         return;

//     r2 = fmax(r2, min_r * min_r);
//     double r = sqrt(r2);

//     // Very simple short-range repulsive force
//     double coef = (1 - cutoff / r) / r2 / mass;
//     particle.ax += coef * dx;
//     particle.ay += coef * dy;
// }

// // Integrate the ODE
// void move(particle_t& p, double size) {
//     // Slightly simplified Velocity Verlet integration
//     // Conserves energy better than explicit Euler method
//     p.vx += p.ax * dt;
//     p.vy += p.ay * dt;
//     p.x += p.vx * dt;
//     p.y += p.vy * dt;

//     // Bounce from walls
//     while (p.x < 0 || p.x > size) {
//         p.x = p.x < 0 ? -p.x : 2 * size - p.x;
//         p.vx = -p.vx;
//     }

//     while (p.y < 0 || p.y > size) {
//         p.y = p.y < 0 ? -p.y : 2 * size - p.y;
//         p.vy = -p.vy;
//     }
// }


// // Put any static global variables here that you will use throughout the simulation.

// void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
// 	// You can use this space to initialize data objects that you may need
// 	// This function will be called once before the algorithm begins
// 	// Do not do any particle simulation here
// }

// void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
//     double subdomain_size = size / num_procs;
//     double x_min = rank * subdomain_size;
//     double x_max = (rank + 1) * subdomain_size;

//     std::vector<particle_t> local_particles;
    
//     // Distribute particles to their respective ranks
//     for (int i = 0; i < num_parts; i++) {
//         if (parts[i].x >= x_min && parts[i].x < x_max) {
//             local_particles.push_back(parts[i]);
//         }
//     }

//     int left = rank - 1;
//     int right = rank + 1;

//     std::vector<particle_t> left_ghost, right_ghost;
//     MPI_Status status;

//     // Send local data and receive ghost particles
//     if (left >= 0) {
//         int send_count = local_particles.size();
//         MPI_Send(&send_count, 1, MPI_INT, left, 0, MPI_COMM_WORLD);
//         MPI_Recv(&send_count, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);

//         left_ghost.resize(send_count);
//         MPI_Sendrecv(local_particles.data(), send_count * sizeof(particle_t), MPI_BYTE, left, 1,
//                      left_ghost.data(), send_count * sizeof(particle_t), MPI_BYTE, left, 1,
//                      MPI_COMM_WORLD, &status);
//     }

//     if (right < num_procs) {
//         int send_count = local_particles.size();
//         MPI_Send(&send_count, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
//         MPI_Recv(&send_count, 1, MPI_INT, right, 0, MPI_COMM_WORLD, &status);

//         right_ghost.resize(send_count);
//         MPI_Sendrecv(local_particles.data(), send_count * sizeof(particle_t), MPI_BYTE, right, 1,
//                      right_ghost.data(), send_count * sizeof(particle_t), MPI_BYTE, right, 1,
//                      MPI_COMM_WORLD, &status);
//     }

//     // Compute forces
//     for (auto& p : local_particles) {
//         p.ax = p.ay = 0;
//         for (auto& neighbor : local_particles) {
//             if (&p != &neighbor) apply_force(p, neighbor);
//         }
//         for (auto& neighbor : left_ghost) {
//             apply_force(p, neighbor);
//         }
//         for (auto& neighbor : right_ghost) {
//             apply_force(p, neighbor);
//         }
//     }

//     // Move particles
//     for (auto& p : local_particles) {
//         move(p, size);
//     }

//     // Gather results back into main array
//     MPI_Allgather(local_particles.data(), local_particles.size() * sizeof(particle_t), MPI_BYTE,
//                   parts, num_parts * sizeof(particle_t), MPI_BYTE, MPI_COMM_WORLD);
// }

// void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
//     // Write this function such that at the end of it, the master (rank == 0)
//     // processor has an in-order view of all particles. That is, the array
//     // parts is complete and sorted by particle id.
//     std::vector<particle_t> all_particles;
//     if (rank == 0) {
//         all_particles.resize(num_parts * num_procs);
//     }

//     MPI_Gather(parts, num_parts * sizeof(particle_t), MPI_BYTE, all_particles.data(), num_parts * sizeof(particle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
//             return a.id < b.id;
//         });
//     }
// }


#include "common.h"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>

// Put any static global variables here that you will use throughout the simulation.
const double grid_step = cutoff*1.00001;
int g_lda;
int g_dims[2], g_coords[2], g_xd, g_yd, g_x0, g_y0;
MPI_Comm g_comm;

struct ParticleContainer;
struct ParticleContainerPredecessor;
struct ParticleContainerPredecessor {
    ParticleContainer* next = nullptr;
};
struct ParticleContainer : ParticleContainerPredecessor {
    ParticleContainerPredecessor* prev = nullptr;
    particle_t* p = nullptr;
    int g_i;
};
enum ParticleKind {
    Owned,
    Ghost
};
ParticleContainerPredecessor* particle_grid;
ParticleContainer* particle_containers;
std::unordered_set<int> g_parts, g_ghosts;

enum Dimension {
    Dim_X = 0,
    Dim_Y = 1,
    Dims = 2
};
enum Direction {
    Dir_Down = 0,
    Dir_Up = 1,
    Dirs = 2
};
struct Neighbor {
    particle_t* send = nullptr;
    particle_t* recv = nullptr;
    int send_count = 0;
    int recv_count = 0;
    MPI_Request send_req, recv_req;
    MPI_Status send_status, recv_status;
    int rank;
    void push(particle_t& p) {
        //if (send != nullptr)
            send[send_count++] = p;
    }
    void begin_send() {
        MPI_Issend(send, send_count, PARTICLE, rank, 0, g_comm, &send_req);
    }
    void finish_send() {
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        send_count = 0;
    }
    void receive() {
        MPI_Probe(rank, MPI_ANY_TAG, g_comm, &recv_status);
        MPI_Get_count(&recv_status, PARTICLE, &recv_count);
        MPI_Recv(recv, recv_count, PARTICLE, rank, MPI_ANY_TAG, g_comm, &recv_status);
    }
};
Neighbor neighbors[Dims][Dirs];


static void inline __attribute__((always_inline))
linkParticle(ParticleContainerPredecessor* pred, ParticleContainer* pc) {
    pc->prev = pred;
    pc->next = pred->next;
    if (pred->next != nullptr)
        pred->next->prev = pc;
    pred->next = pc;
}

static void inline __attribute__((always_inline))
unlinkParticle(ParticleContainer* pc) {
    pc->prev->next = pc->next;
    if (pc->next != nullptr)
        pc->next->prev = pc->prev;
}

// Apply the force from neighbor to particle
static void inline __attribute__((always_inline))
apply_force(particle_t& p, particle_t& p_prime) {
    // Calculate Distance
    double dx = p_prime.x - p.x;
    double dy = p_prime.y - p.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = std::max(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    p.ax += coef * dx;
    p.ay += coef * dy;
    p_prime.ax -= coef * dx;
    p_prime.ay -= coef * dy;
}

static void inline __attribute__((always_inline))
apply_intercell_force(particle_t* const p, const int x_prime, const int y_prime) {
    int i_prime = x_prime+(g_xd+4)*y_prime;
    for(ParticleContainer* pc_prime = particle_grid[i_prime].next; pc_prime != nullptr; pc_prime = pc_prime->next) {
        particle_t* p_prime = pc_prime->p;
        apply_force(*p, *p_prime);
    }
}

// Integrate the ODE
static void inline __attribute__((always_inline))
move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) 
{
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here
    MPI_Dims_create(num_procs, Dims, g_dims);
    int periods[2] = {false, false};
    MPI_Cart_create(MPI_COMM_WORLD, Dims, g_dims, periods, false, &g_comm);
    MPI_Cart_coords(g_comm, rank, Dims, g_coords);
    for (int dim = 0; dim < Dims; ++dim) {
      MPI_Cart_shift(g_comm, dim, 1, &neighbors[dim][0].rank, &neighbors[dim][1].rank);
      for (int dir = 0; dir < Dirs; ++dir) {
        neighbors[dim][dir].send = new particle_t[num_parts];
        neighbors[dim][dir].recv = new particle_t[num_parts];
      }
    }

    g_lda = static_cast<int>(size/grid_step)+1;
    g_xd = (g_lda+g_dims[0]-1)/g_dims[0];
    g_yd = (g_lda+g_dims[1]-1)/g_dims[1];
    g_x0 = g_coords[0]*g_xd;
    g_y0 = g_coords[1]*g_yd;

    particle_grid = new ParticleContainerPredecessor[(g_xd+4)*(g_yd+4)]();
    particle_containers = new ParticleContainer[num_parts]();

    for (int p_i = 0; p_i < num_parts; ++p_i) {
        ParticleContainer* pc = particle_containers+p_i;
        particle_t* p = parts+p_i;
        p->ax = p->ay = 0;
        int g_x = static_cast<int>(p->x / grid_step)-g_x0;
        int g_y = static_cast<int>(p->y / grid_step)-g_y0;
        if (0 <= g_x && g_x < g_xd && 0 <= g_y && g_y < g_yd) {
            g_parts.insert(p_i);
            int g_i = g_x+2+(g_xd+4)*(g_y+2);
            pc->p = p;
            pc->g_i = g_i;
            ParticleContainerPredecessor* g = particle_grid+g_i;
            linkParticle(g, pc);
        }
    }
}

static void inline __attribute__((always_inline))
push_ghosts(
    Dimension dim, Direction dir)
{
    Neighbor& outgoing = neighbors[dim][dir];
    if (outgoing.rank != MPI_PROC_NULL) {
        int x_lo = dim == Dim_X && dir == Dir_Up ? g_xd-1 : 0;
        int x_hi = dim == Dim_X && dir == Dir_Down ? 0 : g_xd-1;
        int y_lo = dim == Dim_Y && dir == Dir_Up ? g_yd-1 : 0;
        int y_hi = dim == Dim_Y && dir == Dir_Down ? 0 : g_yd-1;
        for (int gy = 2+y_lo; gy <= 2+y_hi; ++gy) {
          for (int gx = 2+x_lo; gx <= 2+x_hi; ++gx) {
            int i = gx+(g_xd+4)*gy;
            for (ParticleContainer* pc = particle_grid[i].next; pc != nullptr; pc = pc->next) {
                particle_t* p = pc->p;
                outgoing.push(*p);
            }
          }
        }
    }
}

static void inline __attribute__((always_inline))
neighbor_exchange(
    Dimension dim)
{
    Neighbor& neighbor_down = neighbors[dim][Dir_Down];
    Neighbor& neighbor_up = neighbors[dim][Dir_Up];
    neighbor_down.begin_send();
    neighbor_up.receive();
    neighbor_down.finish_send();
    neighbor_up.begin_send();
    neighbor_down.receive();
    neighbor_up.finish_send();
}

static void inline __attribute__((always_inline))
receive_particles(
    particle_t* parts, ParticleContainer* particle_containers, ParticleContainerPredecessor* particle_grid,
    Dimension dim, Direction dir,
    bool pass_y, ParticleKind kind)
{
    Neighbor& incoming = neighbors[dim][dir];
    if (incoming.rank == MPI_PROC_NULL)
        return;
    for (particle_t* p_recv = incoming.recv; p_recv != incoming.recv+incoming.recv_count; ++p_recv) {
        int p_i = p_recv->id - 1;
        ParticleContainer* pc = particle_containers+p_i;
        particle_t* p = parts+p_i;
        int g_x = static_cast<int>(p_recv->x / grid_step)-g_x0;
        int g_y = static_cast<int>(p_recv->y / grid_step)-g_y0;
        p_recv->ax = p_recv->ay = 0;
        if (pass_y) {
            if (kind != Ghost) {
              if (g_y < 0) {
                neighbors[Dim_Y][Dir_Down].push(*p_recv);
                continue;
              } else if (g_y >= g_yd) {
                neighbors[Dim_Y][Dir_Up].push(*p_recv);
                continue;
              }
            } else {
              if (g_y <= 0) {
                neighbors[Dim_Y][Dir_Down].push(*p_recv);
              } else if (g_y >= g_yd-1) {
                neighbors[Dim_Y][Dir_Up].push(*p_recv);
              }
            }
        }
        if (-1 <= g_x && g_x < g_xd+1 && -1 <= g_y && g_y < g_yd+1) {
            if (0 <= g_x && g_x < g_xd && 0 <= g_y && g_y < g_yd) {
                g_parts.insert(p_i);
            } else if (kind == Ghost) {
                g_ghosts.insert(p_i);
            }
            *p = *p_recv;
            pc->p = p;
            int g_i = g_x+2+(g_xd+4)*(g_y+2);
            pc->g_i = g_i;
            ParticleContainerPredecessor* g = particle_grid+g_i;
            linkParticle(g, pc);
        }
    }
    incoming.recv_count = 0;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) 
{
    // Exchange ghost particles.
    // Step 1: exchange along dimension 0.
    push_ghosts(Dim_X, Dir_Down);
    push_ghosts(Dim_X, Dir_Up);
    neighbor_exchange(Dim_X);
    receive_particles(parts, particle_containers, particle_grid, Dim_X, Dir_Down, true, Ghost);
    receive_particles(parts, particle_containers, particle_grid, Dim_X, Dir_Up, true, Ghost);
    // Step 2: exchange along dimension 1, including incoming particles from step 1.
    push_ghosts(Dim_Y, Dir_Down);
    push_ghosts(Dim_Y, Dir_Up);
    neighbor_exchange(Dim_Y);
    receive_particles(parts, particle_containers, particle_grid, Dim_Y, Dir_Down, false, Ghost);
    receive_particles(parts, particle_containers, particle_grid, Dim_Y, Dir_Up, false, Ghost);
    // Compute Forces
    for (int gy = 1; gy < g_yd+3; ++gy) {
      for (int gx = 1; gx < g_xd+3; ++gx) {
        int i = gx+(g_xd+4)*gy;
        for (ParticleContainer* pc = particle_grid[i].next; pc != nullptr; pc = pc->next) {
            particle_t* p = pc->p;
            apply_intercell_force(p, gx-1, gy-1);
            apply_intercell_force(p, gx, gy-1);
            apply_intercell_force(p, gx+1, gy-1);
            apply_intercell_force(p, gx-1, gy);
//            apply_intercell_force(p, gx, gy);
//            apply_intercell_force(p, gx+1, gy);
//            apply_intercell_force(p, gx-1, gy+1);
//            apply_intercell_force(p, gx, gy+1);
//            apply_intercell_force(p, gx+1, gy+1);
            for (ParticleContainer* pc_prime = pc->next; pc_prime != nullptr; pc_prime = pc_prime->next) apply_force(*p, *pc_prime->p);
        }
      }
    }
    // Move Particles
    for (auto it = g_parts.begin(); it != g_parts.end(); ) {
        int p_i = *it;
        ParticleContainer* pc = particle_containers+p_i;
        particle_t* p = parts+p_i;
        move(*p, size);
            p->ax = p->ay = 0;
        int g_x_prime = static_cast<int>(p->x / grid_step)-g_x0;
        int g_y_prime = static_cast<int>(p->y / grid_step)-g_y0;
        int g_i_prime = g_x_prime+2+(g_xd+4)*(g_y_prime+2);
        if (pc->g_i != g_i_prime) {
            if (g_x_prime < 0 || g_x_prime >= g_xd || g_y_prime < 0 || g_y_prime >= g_yd) {
              unlinkParticle(pc);
              it = g_parts.erase(it);
              if (g_x_prime < 0) {
                neighbors[Dim_X][Dir_Down].push(*p);
              } else if (g_x_prime >= g_xd) {
                neighbors[Dim_X][Dir_Up].push(*p);
              } else if (g_y_prime < 0) {
                neighbors[Dim_Y][Dir_Down].push(*p);
              } else if (g_y_prime >= g_yd) {
                neighbors[Dim_Y][Dir_Up].push(*p);
              }
              continue;
            }
            ParticleContainerPredecessor* g_prime = particle_grid+g_i_prime;
            pc->g_i = g_i_prime;
            unlinkParticle(pc);
            linkParticle(g_prime, pc);
        }
        ++it;
    }
    for (const int p_i: g_ghosts) {
        unlinkParticle(particle_containers+p_i);
    }
    g_ghosts.clear();
    // Exchange moved particles.
    // Step 1: exchange along dimension 0.
    neighbor_exchange(Dim_X);
    receive_particles(parts, particle_containers, particle_grid, Dim_X, Dir_Down, true, Owned);
    receive_particles(parts, particle_containers, particle_grid, Dim_X, Dir_Up, true, Owned);
    // Step 2: exchange along dimension 1, including incoming particles from step 1.
    neighbor_exchange(Dim_Y);
    receive_particles(parts, particle_containers, particle_grid, Dim_Y, Dir_Down, false, Owned);
    receive_particles(parts, particle_containers, particle_grid, Dim_Y, Dir_Up, false, Owned);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
MPI_Barrier(g_comm);
    std::vector<particle_t> send;
    send.reserve(g_parts.size());
    std::vector<particle_t> recv;
    for (const int p_i: g_parts) {
        send.push_back(parts[p_i]);
    }
    int sendcount = send.size();
    int* recvcounts, *recvdispls;
    if (rank == 0) {
        recvcounts = new int[num_procs];
        recvdispls = new int[num_procs];
    }
    MPI_Barrier(g_comm);
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, g_comm);
    if (rank == 0) {
        recvdispls[0] = 0;
        for (int i = 1; i < num_procs; ++i) recvdispls[i] = recvdispls[i-1]+recvcounts[i-1];
        recv.resize(recvdispls[num_procs-1]+recvcounts[num_procs-1]);
    }
    MPI_Barrier(g_comm);
    MPI_Gatherv(send.data(), sendcount, PARTICLE, recv.data(), recvcounts, recvdispls, PARTICLE, 0, g_comm);
    if (rank == 0) {
        for (const auto& p_recv: recv) {
            int p_i = p_recv.id - 1;
            parts[p_i] = p_recv;
        }
    }
}
