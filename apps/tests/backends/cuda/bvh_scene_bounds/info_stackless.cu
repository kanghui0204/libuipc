#include <app/app.h>
#include <collision_detection/info_stackless_bvh.h>
#include "oracle.h"

TEST_CASE("bvh scene bounds: InfoStacklessBVH",
          "[collision detection][bvh_scene_bounds]")
{
    test_bvh_scene_bounds::check_impl<InfoStacklessBVH::Impl>();
}
