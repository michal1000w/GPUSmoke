#ifndef __SIMULATION
#define __SIMULATION

#include "Libraries.h"






// Runs a single iteration of the simulation
void simulate_fluid(fluid_state& state, std::vector<OBJECT>& object_list,
    int ACCURACY_STEPS = 35, bool DEBUG = false, int frame = 0,
    float Dissolve_rate = 0.95f, float Ambient_temp = 0.0f,
    float Diverge_Rate = 0.5f, float Smoke_Buoyancy = 1.0f, float Pressure = -1.0f, float Flame_Dissolve = 0.99f,
    float sscale = 0.7, float sintensity = 1, float soffset = 0.07, bool Upsampling = false, bool UpsamplingVelocity = false,
    bool UpsamplingDensity = false, float time_anim = 0.5, float density_cutoff = 0.01f, float deviceCount = 1,
    float max_velocity = 3.0f, float influence_on_velocity = 0.1f)
{
    float AMBIENT_TEMPERATURE = Ambient_temp;//0.0f
    //float BUOYANCY = buoancy; //1.0f


    
    /////////////GLOBAL EVALUATE//////////////////////
    //Smoke_Buoyancy += 0.5f * cosf(-0.8f * float(state.step));
    //////////////////////////////////////////////////

    unsigned char current_device = 0;


    //cudaStream_t stream1; cudaStreamCreate(&stream1); 
    //cudaStream_t stream2; cudaStreamCreate(&stream2);


    float measured_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //std::cout << "|" << BLOCK_SIZE << ";" << LOC_SIZE << "|";

    const int s = BLOCK_SIZE;//8
    dim3 block(s, s, s);
    
    dim3 grid((state.dim.x + block.x - 1) / block.x,
        (state.dim.y + block.y - 1) / block.y,
        (state.dim.z + block.z - 1) / block.z);

    cudaEventRecord(start, 0);



    current_device = 0;
    cudaSetDevice(current_device);

    advection << <grid, block >> > (
        state.velocity->readTargett(current_device),
        state.velocity->readTargett(current_device),
        state.velocity->writeTargett(current_device),
        state.dim, state.time_step, 1.0);//1.0
    state.velocity->swap();

    cudaMemcpyAsync(state.velocity->readTargett(1), state.velocity->readTargett(0), state.velocity->byteCount(), cudaMemcpyDeviceToDevice);

    std::cout << cudaGetErrorString(cudaGetLastError());


    //current_device = 1;
    cudaSetDevice(1);

    advection << <grid, block >> > (
        state.velocity->readTargett(current_device),
        state.temperature->readTargett(current_device),
        state.temperature->writeTargett(0),
        state.dim, state.time_step, 0.998);//0.998
    state.temperature->swap();


    advection << <grid, block >> > (
        state.velocity->readTargett(current_device),
        state.flame->readTargett(current_device),
        state.flame->writeTargett(0),
        state.dim, state.time_step, Flame_Dissolve);
    state.flame->swap();


    advection << <grid, block >> > (  //zanikanie
        state.velocity->readTargett(current_device),
        state.density->readTargett(current_device),
        state.density->writeTargett(current_device),
        state.dim, state.time_step, Dissolve_rate);//0.995
    state.density->swap();

    cudaSetDevice(0);








    //wznoszenie si� dymu
    buoyancy << <grid, block >> > (
        state.velocity->readTargett(current_device),
        state.temperature->readTargett(current_device),
        state.density->readTargett(current_device),
        state.velocity->writeTargett(current_device),
        AMBIENT_TEMPERATURE, state.time_step, Smoke_Buoyancy, state.f_weight, state.dim); //1.0f
    state.velocity->swap();

    float3 location = state.impulseLoc;


    /////Z - glebia
    /////X - lewo prawo
    /////Y - gora dol
    float MOVEMENT_SIZE = 3.0;//9
    float MOVEMENT_SPEED = 20;//10
    bool MOVEMENT = true;



    


    if (DEBUG || true)
    for (int i = 0; i < object_list.size(); i++) {
        OBJECT current = object_list[i];
        
        
        //INITIAL ANIMATION
        if (MOVEMENT && current.get_type() != "explosion") {
            if (EXAMPLE__ == 1) {
                object_list[i].set_location(current.get_location().x + MOVEMENT_SIZE * 2.0 * sinf(-0.04f * MOVEMENT_SPEED * float(state.step)),
                    current.get_location().y + cosf(-0.03f * float(state.step)),
                    current.get_location().z + MOVEMENT_SIZE * cosf(-0.02f * MOVEMENT_SPEED * float(state.step))
                );
            }
            else if (EXAMPLE__ == 2) {
                float MOVEMENT_SIZE = 9.0;//9
                float MOVEMENT_SPEED = 0.7;//10
                object_list[i].set_location(current.get_location().x + MOVEMENT_SIZE * sinf(-0.12786786f * MOVEMENT_SPEED * float(state.step)),
                    current.get_location().y + MOVEMENT_SIZE * 0.05f * cosf(-0.03f * float(state.step)),
                    current.get_location().z + MOVEMENT_SIZE * cosf(-0.0767637f * MOVEMENT_SPEED * float(state.step))
                );
            }
        }





        //REAL ANIMATION
        if (current.get_type() == "explosion") { // RESCALE
            if (frame >= current.frame_range_min && frame <= current.frame_range_max) {
                /*
                    */
                float direction = 1.0f;
                resize_sphere_vel << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), object_list[i].size, direction, max_velocity, influence_on_velocity,
                    state.dim
                    );
                object_list[i].size += direction;
                //std::cout << cudaGetErrorString(cudaGetLastError());
            }
        }






        //GIVING THE FLOW
        float3 SIZEE = make_float3(current.get_size(), current.get_size(), current.get_size());


        if (current.get_type() == "emitter") {
            wavey_impulse_temperature_new << < grid, block >> > (
                state.temperature->readTargett(current_device),
                state.velocity->readTargett(current_device),
                current.get_location(), SIZEE,
                current.get_impulseTemp(), current.get_initial_velocity(), current.get_velocity_frequence(),
                state.dim,
                frame
                );
            wavey_impulse_density_new << < grid, block >> > (
                state.density->readTargett(current_device),
                current.get_location(), SIZEE,
                current.get_impulseDensity(), 1.0f, current.get_velocity_frequence(),
                state.dim,
                frame
                );
        }
        else if (current.get_type() == "smoke"){
            impulse << <grid, block >> > (
                state.temperature->readTargett(current_device),
                current.get_location(), current.get_size(),
                current.get_impulseTemp(),
                state.dim
                );
            impulse << <grid, block >> > (
                state.density->readTargett(current_device),
                current.get_location(), current.get_size(),
                current.get_impulseDensity(),
                state.dim
                );
        }
        else if (current.get_type() == "explosion") {
            if (frame >= current.frame_range_min && frame <= current.frame_range_max) {
                impulse << <grid, block >> > (
                    state.flame->readTargett(current_device),
                    current.get_location(), current.get_size(),
                    current.get_impulseTemp(),
                    state.dim
                    );
                impulse << <grid, block >> > (
                    state.temperature->readTargett(current_device),
                    current.get_location(), current.get_size(),
                    current.get_impulseTemp(),
                    state.dim
                    );
                impulse << <grid, block >> > (
                    state.density->readTargett(current_device),
                    current.get_location(), current.get_size(),
                    current.get_impulseDensity(),
                    state.dim
                    );
            }
        }
        /*
        else if (current.get_type() == "vdb") {
            impulse_vdb << <grid, block >> > (
                state.temperature->readTarget(),
                current.get_location(),
                current.get_impulseTemp(),
                state.dim,
                current.get_density_grid().get_grid_device_temp(),
                current.get_initial_temp()
                );
            impulse_vdb << <grid, block >> > (
                state.density->readTarget(),
                current.get_location(),
                current.get_impulseDensity(),
                state.dim
                ,current.get_density_grid().get_grid_device()
                );
        }
        else if (current.get_type() == "vdbs") {
            impulse_vdb_single << <grid, block >> > (
                state.temperature->readTarget(),
                current.get_location(),
                current.get_impulseTemp(),
                state.dim,
                current.get_density_grid().get_grid_device_temp(),
                current.get_initial_temp()
                );
            impulse_vdb_single << <grid, block >> > (
                state.density->readTarget(),
                current.get_location(),
                current.get_impulseDensity(),
                state.dim
                , current.get_density_grid().get_grid_device()
                );
        }
        */
        else if (current.get_type() == "fff") {
            if (current.square) {
                force_field_force << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength * current.force_strength,
                    state.dim
                    );
            }
            else {
                force_field_force << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength,
                    state.dim
                    );
            }
        }
        else if (current.get_type() == "ffp") {
            if (current.square) {
                force_field_power << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength * current.force_strength,
                    state.dim
                    );
            }
            else {
                force_field_power << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength,
                    state.dim
                    );
            }
        }
        else if (current.get_type() == "fft") {
            if (current.square) {
                force_field_turbulance << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength*current.force_strength, current.set_vel_freq + current.velocity_frequence,
                    state.dim,
                    frame
                    );
            }
            else {
                force_field_turbulance << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength, current.set_vel_freq + current.velocity_frequence,
                    state.dim,
                    frame
                    );
            }
            if (current.vel_freq_mov) {
                current.set_vel_freq += current.vel_freq_step;
                if (current.set_vel_freq >= current.max_vel_freq)
                    current.vel_freq_mov = false;
            }
            else {
                current.set_vel_freq -= current.vel_freq_step;
                if (current.set_vel_freq <= 0.0)
                    current.vel_freq_mov = true;
            }
        }
        else if (current.get_type() == "ffw") {
            if (current.square) {
                float3 direction = make_float3(current.force_direction[0], current.force_direction[1], current.force_direction[2]);
                force_field_wind << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength * current.force_strength,
                    direction,
                    state.dim
                    );
            }
            else {
                float3 direction = make_float3(current.force_direction[0], current.force_direction[1], current.force_direction[2]);
                force_field_wind << < grid, block >> > (
                    state.velocity->readTargett(current_device),
                    current.get_location(), current.size,
                    current.force_strength,
                    direction,
                    state.dim
                    );
            }
        }
        else if (current.get_type() == "cols") {
            /*
            collision_sphere << < grid, block >> > (
                state.velocity->readTarget(),
                current.get_location(), current.size,
                state.dim
                );
            */
            collision_sphere2 << <grid, block >> > (
                state.velocity->readTargett(current_device),
                state.temperature->readTargett(current_device),
                state.density->readTargett(current_device),
                state.dim, current.get_location(), current.size,
                AMBIENT_TEMPERATURE);
        }

        //if (false)
        if (current.get_type() == "smoke" || current.get_type() == "vdbs") {
            current.cudaFree();
            object_list.erase(object_list.begin() + i); //remove emitter from the list
        }
    }

    //environment turbulance
    divergence << <grid, block >> > (
        state.velocity->readTargett(current_device),
        state.diverge[0], state.dim, Diverge_Rate);//0.5
    /*
        */




// clear pressure
    impulse << <grid, block >> > (
        state.pressure->readTargett(current_device),
        make_float3(0.0), 1000000.0f,
        0.0f, state.dim);



    for (int i = 0; i < ACCURACY_STEPS; i++)
    {
        pressure_solve << <grid, block >> > (
            state.diverge[0],
            state.pressure->readTargett(current_device),
            state.pressure->writeTargett(current_device),
            state.dim, Pressure); //-1.0
        state.pressure->swap();
    }

    subtract_pressure << <grid, block >> > (
        state.velocity->readTargett(current_device),
        state.velocity->writeTargett(current_device),
        state.pressure->readTargett(current_device),
        state.dim, -1.0f * Pressure);//1.0
        //state.dim, 1.0f);//1.0
    state.velocity->swap();
    /*
    */




    if (Upsampling) {
        //BETA
        for (int i = 0; i < 1; i++) {
            if (UpsamplingDensity) {
                applyNoiseDT << <grid, block >> > (
                    state.temperature->readTargett(current_device),
                    state.density->readTargett(current_device),
                    state.temperature->writeTargett(current_device),
                    state.density->writeTargett(current_device),
                    state.noise->readTargett(current_device),
                    state.dim,
                    /*intensity=0.45f*/sintensity,
                    /*offset=0.075f*/soffset,
                    /*scale=0.7*/sscale,
                    frame,
                    time_anim,
                    density_cutoff
                    );
                state.temperature->swap();
                state.density->swap();
            }

            if (UpsamplingVelocity) {
                applyNoiseV << <grid, block >> > (
                    state.velocity->readTargett(current_device),
                    state.velocity->writeTargett(current_device),
                    state.noise->readTargett(current_device),
                    state.dim,
                    /*intensity=0.45f*/sintensity,
                    /*offset=0.075f*/soffset,
                    /*scale=0.7*/sscale,
                    frame,
                    time_anim);
                state.velocity->swap();
            }
        }
    }
    


    cudaThreadSynchronize();
    checkCudaErrors(cudaMemcpyAsync(state.density->readTargett(0), state.density->readTargett(1), state.density->byteCount(), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(state.flame->readTargett(0), state.flame->readTargett(1), state.flame->byteCount(), cudaMemcpyDeviceToDevice));
    current_device = 0;
    cudaSetDevice(0);


    cudaEventRecord(stop, 0);
    cudaThreadSynchronize();
    cudaEventElapsedTime(&measured_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Simulation Time: " << measured_time << " ||";
}






#endif