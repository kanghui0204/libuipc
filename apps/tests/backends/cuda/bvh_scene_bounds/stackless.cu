#include <app/app.h>
#include <collision_detection/stackless_bvh.h>
#include "oracle.h"

TEST_CASE("bvh scene bounds: StacklessBVH",
          "[collision detection][bvh_scene_bounds]")
{
    test_bvh_scene_bounds::check_impl<StacklessBVH::Impl>();
}
