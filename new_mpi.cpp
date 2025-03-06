#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <unordered_set>

// Simulation parameters and grid setup.
const double CELL_SIZE = cutoff * 1.00001;
int GRID_DIMENSIONS[2], LOCAL_COORDINATES[2], LOCAL_GRID_WIDTH, LOCAL_GRID_HEIGHT, GLOBAL_X_OFFSET, GLOBAL_Y_OFFSET;
MPI_Comm WORLD_COMM;

// Particle management structures.
struct ParticleCell;
struct ParticleCellPrevious;
struct ParticleCellPrevious {
    ParticleCell* next = nullptr;
};
struct ParticleCell : ParticleCellPrevious {
    ParticleCellPrevious* prev = nullptr;
    particle_t* particle = nullptr;
    int grid_index;
};

enum ParticleType {
    OWNED,
    GHOST
};

ParticleCellPrevious* grid_cells;
ParticleCell* particle_cells;
std::unordered_set<int> owned_particles, ghost_particles;

// Neighbor communication structures.
enum Axis {
    X_AXIS = 0,
    Y_AXIS = 1,
    AXES = 2
};
enum Direction {
    NEGATIVE = 0,
    POSITIVE = 1,
    DIRECTIONS = 2
};

struct NeighborBuffer {
    particle_t* send_buffer = nullptr;
    particle_t* receive_buffer = nullptr;
    int send_count = 0;
    int receive_count = 0;
    MPI_Request send_request, receive_request;
    MPI_Status send_status, receive_status;
    int neighbor_rank;

    void add_particle(const particle_t& p) {
        send_buffer[send_count++] = p;
    }

    void initiate_send() {
        MPI_Issend(send_buffer, send_count, PARTICLE, neighbor_rank, 0, WORLD_COMM, &send_request);
    }

    void complete_send() {
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        send_count = 0;
    }

    void receive_particles() {
        MPI_Probe(neighbor_rank, MPI_ANY_TAG, WORLD_COMM, &receive_status);
        MPI_Get_count(&receive_status, PARTICLE, &receive_count);
        MPI_Recv(receive_buffer, receive_count, PARTICLE, neighbor_rank, MPI_ANY_TAG, WORLD_COMM, &receive_status);
    }
};

NeighborBuffer neighbor_buffers[AXES][DIRECTIONS];

// Helper functions for particle cell management.
static void inline link_particle_to_cell(ParticleCellPrevious* cell_previous, ParticleCell* cell) {
    cell->prev = cell_previous;
    cell->next = cell_previous->next;
    if (cell_previous->next != nullptr)
        cell_previous->next->prev = cell;
    cell_previous->next = cell;
}

static void inline unlink_particle_from_cell(ParticleCell* cell) {
    cell->prev->next = cell->next;
    if (cell->next != nullptr)
        cell->next->prev = cell->prev;
}

// Physics functions.
static void inline apply_particle_interaction(particle_t& p1, particle_t& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance_squared = dx * dx + dy * dy;

    if (distance_squared > cutoff * cutoff)
        return;

    distance_squared = std::max(distance_squared, min_r * min_r);
    double distance = sqrt(distance_squared);

    double interaction_coefficient = (1 - cutoff / distance) / distance_squared / mass;
    p1.ax += interaction_coefficient * dx;
    p1.ay += interaction_coefficient * dy;
    p2.ax -= interaction_coefficient * dx;
    p2.ay -= interaction_coefficient * dy;
}

static void inline apply_cell_interactions(particle_t* particle, int neighbor_x, int neighbor_y) {
    int cell_index = neighbor_x + (LOCAL_GRID_WIDTH + 4) * neighbor_y;
    for (ParticleCell* neighbor_cell = grid_cells[cell_index].next; neighbor_cell != nullptr; neighbor_cell = neighbor_cell->next) {
        particle_t* neighbor_particle = neighbor_cell->particle;
        apply_particle_interaction(*particle, *neighbor_particle);
    }
}

static void inline update_particle_position(particle_t& particle, double simulation_size) {
    particle.vx += particle.ax * dt;
    particle.vy += particle.ay * dt;
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;

    while (particle.x < 0 || particle.x > simulation_size) {
        particle.x = particle.x < 0 ? -particle.x : 2 * simulation_size - particle.x;
        particle.vx = -particle.vx;
    }

    while (particle.y < 0 || particle.y > simulation_size) {
        particle.y = particle.y < 0 ? -particle.y : 2 * simulation_size - particle.y;
        particle.vy = -particle.vy;
    }
}

// Initialization and simulation functions.
void init_simulation(particle_t* particles, int particle_count, double simulation_size, int process_rank, int process_count) {
    MPI_Dims_create(process_count, AXES, GRID_DIMENSIONS);
    int periodic_boundaries[2] = {false, false};
    MPI_Cart_create(MPI_COMM_WORLD, AXES, GRID_DIMENSIONS, periodic_boundaries, false, &WORLD_COMM);
    MPI_Cart_coords(WORLD_COMM, process_rank, AXES, LOCAL_COORDINATES);

    for (int axis = 0; axis < AXES; ++axis) {
        MPI_Cart_shift(WORLD_COMM, axis, 1, &neighbor_buffers[axis][NEGATIVE].neighbor_rank, &neighbor_buffers[axis][POSITIVE].neighbor_rank);
        for (int direction = 0; direction < DIRECTIONS; ++direction) {
            neighbor_buffers[axis][direction].send_buffer = new particle_t[particle_count];
            neighbor_buffers[axis][direction].receive_buffer = new particle_t[particle_count];
        }
    }

    int global_grid_size = static_cast<int>(simulation_size / CELL_SIZE) + 1;
    LOCAL_GRID_WIDTH = (global_grid_size + GRID_DIMENSIONS[0] - 1) / GRID_DIMENSIONS[0];
    LOCAL_GRID_HEIGHT = (global_grid_size + GRID_DIMENSIONS[1] - 1) / GRID_DIMENSIONS[1];
    GLOBAL_X_OFFSET = LOCAL_COORDINATES[0] * LOCAL_GRID_WIDTH;
    GLOBAL_Y_OFFSET = LOCAL_COORDINATES[1] * LOCAL_GRID_HEIGHT;

    grid_cells = new ParticleCellPrevious[(LOCAL_GRID_WIDTH + 4) * (LOCAL_GRID_HEIGHT + 4)]();
    particle_cells = new ParticleCell[particle_count]();

    for (int particle_index = 0; particle_index < particle_count; ++particle_index) {
        ParticleCell* cell = particle_cells + particle_index;
        particle_t* particle = particles + particle_index;
        particle->ax = particle->ay = 0;

        int global_x = static_cast<int>(particle->x / CELL_SIZE) - GLOBAL_X_OFFSET;
        int global_y = static_cast<int>(particle->y / CELL_SIZE) - GLOBAL_Y_OFFSET;

        if (0 <= global_x && global_x < LOCAL_GRID_WIDTH && 0 <= global_y && global_y < LOCAL_GRID_HEIGHT) {
            owned_particles.insert(particle_index);
            int local_index = global_x + 2 + (LOCAL_GRID_WIDTH + 4) * (global_y + 2);
            cell->particle = particle;
            cell->grid_index = local_index;
            ParticleCellPrevious* grid_cell = grid_cells + local_index;
            link_particle_to_cell(grid_cell, cell);
        }
    }
}

// ... (rest of the code - push_ghosts, neighbor_exchange, receive_particles, simulate_one_step, gather_for_save)
static void inline push_ghosts(Axis axis, Direction direction) {
    NeighborBuffer& buffer = neighbor_buffers[axis][direction];
    if (buffer.neighbor_rank != MPI_PROC_NULL) {
        int x_start = (axis == X_AXIS && direction == POSITIVE) ? LOCAL_GRID_WIDTH - 1 : 0;
        int x_end = (axis == X_AXIS && direction == NEGATIVE) ? 0 : LOCAL_GRID_WIDTH - 1;
        int y_start = (axis == Y_AXIS && direction == POSITIVE) ? LOCAL_GRID_HEIGHT - 1 : 0;
        int y_end = (axis == Y_AXIS && direction == NEGATIVE) ? 0 : LOCAL_GRID_HEIGHT - 1;

        for (int y = 2 + y_start; y <= 2 + y_end; ++y) {
            for (int x = 2 + x_start; x <= 2 + x_end; ++x) {
                int index = x + (LOCAL_GRID_WIDTH + 4) * y;
                for (ParticleCell* cell = grid_cells[index].next; cell != nullptr; cell = cell->next) {
                    buffer.add_particle(*(cell->particle));
                }
            }
        }
    }
}

static void inline neighbor_exchange(Axis axis) {
    NeighborBuffer& negative_buffer = neighbor_buffers[axis][NEGATIVE];
    NeighborBuffer& positive_buffer = neighbor_buffers[axis][POSITIVE];

    negative_buffer.initiate_send();
    positive_buffer.receive_particles();
    negative_buffer.complete_send();

    positive_buffer.initiate_send();
    negative_buffer.receive_particles();
    positive_buffer.complete_send();
}

static void inline receive_particles(particle_t* particles, ParticleCell* particle_cells, ParticleCellPrevious* grid_cells, Axis axis, Direction direction, bool pass_y, ParticleType type) {
    NeighborBuffer& buffer = neighbor_buffers[axis][direction];
    if (buffer.neighbor_rank == MPI_PROC_NULL) return;

    for (particle_t* received_particle = buffer.receive_buffer; received_particle < buffer.receive_buffer + buffer.receive_count; ++received_particle) {
        int particle_index = received_particle->id - 1;
        ParticleCell* cell = particle_cells + particle_index;
        particle_t* particle = particles + particle_index;

        int global_x = static_cast<int>(received_particle->x / CELL_SIZE) - GLOBAL_X_OFFSET;
        int global_y = static_cast<int>(received_particle->y / CELL_SIZE) - GLOBAL_Y_OFFSET;

        received_particle->ax = received_particle->ay = 0;

        if (pass_y) {
            if (type != GHOST) {
                if (global_y < 0) {
                    neighbor_buffers[Y_AXIS][NEGATIVE].add_particle(*received_particle);
                    continue;
                } else if (global_y >= LOCAL_GRID_HEIGHT) {
                    neighbor_buffers[Y_AXIS][POSITIVE].add_particle(*received_particle);
                    continue;
                }
            } else {
                if (global_y <= 0) {
                    neighbor_buffers[Y_AXIS][NEGATIVE].add_particle(*received_particle);
                } else if (global_y >= LOCAL_GRID_HEIGHT - 1) {
                    neighbor_buffers[Y_AXIS][POSITIVE].add_particle(*received_particle);
                }
            }
        }

        if (-1 <= global_x && global_x < LOCAL_GRID_WIDTH + 1 && -1 <= global_y && global_y < LOCAL_GRID_HEIGHT + 1) {
            if (0 <= global_x && global_x < LOCAL_GRID_WIDTH && 0 <= global_y && global_y < LOCAL_GRID_HEIGHT) {
                owned_particles.insert(particle_index);
            } else if (type == GHOST) {
                ghost_particles.insert(particle_index);
            }

            *particle = *received_particle;
            cell->particle = particle;
            int local_index = global_x + 2 + (LOCAL_GRID_WIDTH + 4) * (global_y + 2);
            cell->grid_index = local_index;
            ParticleCellPrevious* grid_cell = grid_cells + local_index;
            link_particle_to_cell(grid_cell, cell);
        }
    }
    buffer.receive_count = 0;
}

void simulate_one_step(particle_t* particles, int particle_count, double simulation_size, int process_rank, int process_count) {
    // Exchange ghost particles.
    // Step 1: exchange along dimension 0.
    push_ghosts(X_AXIS, NEGATIVE);
    push_ghosts(X_AXIS, POSITIVE);
    neighbor_exchange(X_AXIS);
    receive_particles(particles, particle_cells, grid_cells, X_AXIS, NEGATIVE, true, GHOST);
    receive_particles(particles, particle_cells, grid_cells, X_AXIS, POSITIVE, true, GHOST);

    // Step 2: exchange along dimension 1, including incoming particles from step 1.
    push_ghosts(Y_AXIS, NEGATIVE);
    push_ghosts(Y_AXIS, POSITIVE);
    neighbor_exchange(Y_AXIS);
    receive_particles(particles, particle_cells, grid_cells, Y_AXIS, NEGATIVE, false, GHOST);
    receive_particles(particles, particle_cells, grid_cells, Y_AXIS, POSITIVE, false, GHOST);

    // Compute Forces
    for (int y = 1; y < LOCAL_GRID_HEIGHT + 3; ++y) {
        for (int x = 1; x < LOCAL_GRID_WIDTH + 3; ++x) {
            int index = x + (LOCAL_GRID_WIDTH + 4) * y;
            for (ParticleCell* cell = grid_cells[index].next; cell != nullptr; cell = cell->next) {
                particle_t* particle = cell->particle;
                apply_cell_interactions(particle, x - 1, y - 1);
                apply_cell_interactions(particle, x, y - 1);
                apply_cell_interactions(particle, x + 1, y - 1);
                apply_cell_interactions(particle, x - 1, y);
                //apply_cell_interactions(particle, x, y);
                //apply_cell_interactions(particle, x + 1, y);
                apply_cell_interactions(particle, x - 1, y + 1);
                //apply_cell_interactions(particle, x, y + 1);
                //apply_cell_interactions(particle, x + 1, y + 1);
                for (ParticleCell* neighbor_cell = cell->next; neighbor_cell != nullptr; neighbor_cell = neighbor_cell->next) {
                  apply_particle_interaction(*particle, *(neighbor_cell->particle));
                }
            }
        }
    }

    // Move Particles
    for (auto it = owned_particles.begin(); it != owned_particles.end();) {
        int particle_index = *it;
        ParticleCell* cell = particle_cells + particle_index;
        particle_t* particle = particles + particle_index;
        update_particle_position(*particle, simulation_size);
        particle->ax = particle->ay = 0;

        int global_x_new = static_cast<int>(particle->x / CELL_SIZE) - GLOBAL_X_OFFSET;
        int global_y_new = static_cast<int>(particle->y / CELL_SIZE) - GLOBAL_Y_OFFSET;
        int local_index_new = global_x_new + 2 + (LOCAL_GRID_WIDTH + 4) * (global_y_new + 2);

        if (cell->grid_index != local_index_new) {
            if (global_x_new < 0 || global_x_new >= LOCAL_GRID_WIDTH || global_y_new < 0 || global_y_new >= LOCAL_GRID_HEIGHT) {
                unlink_particle_from_cell(cell);
                it = owned_particles.erase(it);

                if (global_x_new < 0) {
                    neighbor_buffers[X_AXIS][NEGATIVE].add_particle(*particle);
                } else if (global_x_new >= LOCAL_GRID_WIDTH) {
                    neighbor_buffers[X_AXIS][POSITIVE].add_particle(*particle);
                } else if (global_y_new < 0) {
                    neighbor_buffers[Y_AXIS][NEGATIVE].add_particle(*particle);
                } else if (global_y_new >= LOCAL_GRID_HEIGHT) {
                    neighbor_buffers[Y_AXIS][POSITIVE].add_particle(*particle);
                }
                continue;
            }
            ParticleCellPrevious* grid_cell_new = grid_cells + local_index_new;
            cell->grid_index = local_index_new;
            unlink_particle_from_cell(cell);
            link_particle_to_cell(grid_cell_new, cell);
        }
        ++it;
    }

    for (const int particle_index : ghost_particles) {
        unlink_particle_from_cell(particle_cells + particle_index);
    }
    ghost_particles.clear();

    // Exchange moved particles.
    // Step 1: exchange along dimension 0.
    neighbor_exchange(X_AXIS);
    receive_particles(particles, particle_cells, grid_cells, X_AXIS, NEGATIVE, true, OWNED);
    receive_particles(particles, particle_cells, grid_cells, X_AXIS, POSITIVE, true, OWNED);

    // Step 2: exchange along dimension 1, including incoming particles from step 1.
    neighbor_exchange(Y_AXIS);
    receive_particles(particles, particle_cells, grid_cells, Y_AXIS, NEGATIVE, false, OWNED);
    receive_particles(particles, particle_cells, grid_cells, Y_AXIS, POSITIVE, false, OWNED);
}


void gather_for_save(particle_t* particles, int particle_count, double simulation_size, int process_rank, int process_count) {
    MPI_Barrier(WORLD_COMM);

    std::vector<particle_t> send_particles;
    send_particles.reserve(owned_particles.size());

    for (const int particle_index : owned_particles) {
        send_particles.push_back(particles[particle_index]);
    }

    int send_count = send_particles.size();
    int* receive_counts = nullptr;
    int* receive_displacements = nullptr;
    std::vector<particle_t> received_particles;

    if (process_rank == 0) {
        receive_counts = new int[process_count];
        receive_displacements = new int[process_count];
    }

    MPI_Barrier(WORLD_COMM);
    MPI_Gather(&send_count, 1, MPI_INT, receive_counts, 1, MPI_INT, 0, WORLD_COMM);

    if (process_rank == 0) {
        receive_displacements[0] = 0;
        for (int i = 1; i < process_count; ++i) {
            receive_displacements[i] = receive_displacements[i - 1] + receive_counts[i - 1];
        }
        int total_received = receive_displacements[process_count - 1] + receive_counts[process_count - 1];
        received_particles.resize(total_received);
    }

    MPI_Barrier(WORLD_COMM);
    MPI_Gatherv(send_particles.data(), send_count, PARTICLE, received_particles.data(), receive_counts, receive_displacements, PARTICLE, 0, WORLD_COMM);

    if (process_rank == 0) {
        for (const auto& received_particle : received_particles) {
            int original_index = received_particle.id - 1;
            particles[original_index] = received_particle;
        }
    }

    if (process_rank == 0) {
        delete[] receive_counts;
        delete[] receive_displacements;
    }
}
