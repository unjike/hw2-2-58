#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
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


// Put any static global variables here that you will use throughout the simulation.

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double subdomain_size = size / num_procs;
    double x_min = rank * subdomain_size;
    double x_max = (rank + 1) * subdomain_size;

    std::vector<particle_t> local_particles;
    
    // Distribute particles to their respective ranks
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].x >= x_min && parts[i].x < x_max) {
            local_particles.push_back(parts[i]);
        }
    }

    int left = rank - 1;
    int right = rank + 1;

    std::vector<particle_t> left_ghost, right_ghost;
    MPI_Status status;

    // Send local data and receive ghost particles
    if (left >= 0) {
        int send_count = local_particles.size();
        MPI_Send(&send_count, 1, MPI_INT, left, 0, MPI_COMM_WORLD);
        MPI_Recv(&send_count, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);

        left_ghost.resize(send_count);
        MPI_Sendrecv(local_particles.data(), send_count * sizeof(particle_t), MPI_BYTE, left, 1,
                     left_ghost.data(), send_count * sizeof(particle_t), MPI_BYTE, left, 1,
                     MPI_COMM_WORLD, &status);
    }

    if (right < num_procs) {
        int send_count = local_particles.size();
        MPI_Send(&send_count, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
        MPI_Recv(&send_count, 1, MPI_INT, right, 0, MPI_COMM_WORLD, &status);

        right_ghost.resize(send_count);
        MPI_Sendrecv(local_particles.data(), send_count * sizeof(particle_t), MPI_BYTE, right, 1,
                     right_ghost.data(), send_count * sizeof(particle_t), MPI_BYTE, right, 1,
                     MPI_COMM_WORLD, &status);
    }

    // Compute forces
    for (auto& p : local_particles) {
        p.ax = p.ay = 0;
        for (auto& neighbor : local_particles) {
            if (&p != &neighbor) apply_force(p, neighbor);
        }
        for (auto& neighbor : left_ghost) {
            apply_force(p, neighbor);
        }
        for (auto& neighbor : right_ghost) {
            apply_force(p, neighbor);
        }
    }

    // Move particles
    for (auto& p : local_particles) {
        move(p, size);
    }

    // Gather results back into main array
    MPI_Allgather(local_particles.data(), local_particles.size() * sizeof(particle_t), MPI_BYTE,
                  parts, num_parts * sizeof(particle_t), MPI_BYTE, MPI_COMM_WORLD);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    std::vector<particle_t> all_particles;
    if (rank == 0) {
        all_particles.resize(num_parts * num_procs);
    }

    MPI_Gather(parts, num_parts * sizeof(particle_t), MPI_BYTE, all_particles.data(), num_parts * sizeof(particle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });
    }
}
