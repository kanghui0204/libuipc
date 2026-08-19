#include <app/app.h>
#include <collision_detection/info_stackless_bvh_v0.h>
#include "oracle.h"

TEST_CASE("bvh scene bounds: InfoStacklessBVHV0",
          "[collision detection][bvh_scene_bounds]")
{
    test_bvh_scene_bounds::check_impl<InfoStacklessBVHV0::Impl>();
}
