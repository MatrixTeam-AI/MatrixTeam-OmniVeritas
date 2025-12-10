import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import siteConfig from '../../content/site.json';

const SITE_TITLE = siteConfig.title;
const SITE_DESCRIPTION = siteConfig.description;

const BASE_URL = import.meta.env.BASE_URL;
export async function GET(context) {
	const posts = await getCollection('blog');
	return rss({
		title: SITE_TITLE,
		description: SITE_DESCRIPTION,
		site: context.site,
		items: posts.map((post) => ({
			...post.data,
			link: `${BASE_URL}/blog/${post.slug}/`,
		})),
	});
}
